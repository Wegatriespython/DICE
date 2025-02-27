using Distributions, Plots, Optim
using Interpolations
using JLD2
using FileIO

include("TechPortfolio.jl")  # New module for technology portfolio decisions
include("Reporting.jl")
using .TechPortfolio

# -----------------------------------------------------------------------------------
# Model core functions (taken from the legacy model)
# -----------------------------------------------------------------------------------

# CRRA utility function
function u(c, η)
    return c^(1-η)/(1-η)
end

# Compute labor force at time t
compute_L(t, L0, n) = L0 * (1 + n)^t

# Cobb-Douglas production function
compute_Y(A, K, L) = A * K^0.3 * L^0.7

# Compute climate damage factor (using a quadratic damage function)
compute_damage(T, θ1, θ2; σ=0.0) = 1 - (θ1*T + θ2*T^2)

# Update capital stock given production and consumption
update_K(K_prev, Y, C, δ) = (1 - δ)*K_prev + Y - C

# -----------------------------------------------------------------------------------
# Parameters structure
# -----------------------------------------------------------------------------------
Base.@kwdef struct Params
    # Economic parameters
    ρ = 0.03                # Time preference rate
    η = 1.5                 # Elasticity of marginal utility
    T_plan = 50             # Planning horizon (years)
    δ = 0.1                 # Capital depreciation rate
    L0 = 7403               # Labor (million workers)
    n = 0.001               # Labor growth rate
    
    # Climate parameters
    M0 = 851.0              # Initial atmospheric CO2 (GtC)
    α = 3.0                 # Climate sensitivity (°C per doubling CO2)
    
    # Damage parameters
    θ1 = 0.00236            # Linear damage coefficient
    θ2 = 0.0000035          # Quadratic damage coefficient
    σ = 0.0                 # Damage volatility (deterministic case)
    
    # Note: portfolio here is not fixed but will become a decision.
    # (w_Agriculture, w_Manufacturing), with Services weight = 1 - (w_Agriculture + w_Manufacturing)
    portfolio = (0.33, 0.33)
end

# -----------------------------------------------------------------------------------
# Constants for the carbon cycle (kept the same for simplicity)
# -----------------------------------------------------------------------------------
const phi_atm = 0.2    # Fraction remaining in atmosphere
const phi_up  = 0.32   # Fraction going to the upper ocean

# -----------------------------------------------------------------------------------
# State transition for the DP model.
#
# Given state (K, M) at time t, consumption c and portfolio decision (w_Agriculture, w_Manufacturing),
# we compute:
#   - Effective productivity and emission intensity via TechPortfolio.compute_portfolio_metrics.
#   - Production adjusted for climate damages.
#   - Next period's capital and atmospheric CO2.
#
# Returns: (K_new, M_new, immediate_reward)
# -----------------------------------------------------------------------------------
function dp_transition(p::Params, t::Int, K, M, c, portfolio)
    # Compute labor at time t
    L_t = compute_L(t, p.L0, p.n)
    
    # Get portfolio-based technology metrics
    tech = compute_portfolio_metrics(portfolio, t)
    A_t = tech.A
    E_intensity = tech.E
    
    # Climate damage calculation based on current M
    T_temp = p.α * log(M/p.M0) / log(2.0)
    D_val = compute_damage(T_temp, p.θ1, p.θ2; σ=p.σ)
    
    # Production (adjusted for damages)
    Y = compute_Y(A_t, K, L_t) * D_val
    
    # State transition for capital
    K_new = update_K(K, Y, c, p.δ)
    
    # Update emissions cumulatively.
    E = E_intensity * K^(0.4)
    ΔM = E / 3.666   # additional GtC added to the atmosphere
    M_new = M + ΔM
    
    # Immediate utility from consuming c
    immediate_reward = u(c, p.η)
    return K_new, M_new, immediate_reward
end

# -----------------------------------------------------------------------------------
# Dynamic Programming Solver
#
# We discretize the state space over capital (K) and atmospheric CO2 (M) and also
# discretize the control options for consumption c and portfolio decision.
#
# For each grid point and period t we solve:
#
#   V_t(K, M) = max_{c, portfolio} { u(c) + β * V_{t+1}(K', M') }
#
# subject to:
#   K' = (1-δ)*K + Y - c   and
#   M' = M * phi_atm + [E_intensity * K^0.4] / 3.666 * (1 - phi_up)
#
# -----------------------------------------------------------------------------------
function dynamic_program_solver(; kwargs...)
    p = Params(; kwargs...)
    T = p.T_plan
    beta = 1/(1 + p.ρ)
    
    # Define grids for state variables as ranges.
    nK = 20
    nM = 20
    K_min, K_max = 50.0, 1000.0
    M_min, M_max = 500.0, 2000.0
    K_grid = range(K_min, K_max, length=nK)
    M_grid = range(M_min, M_max, length=nM)
    
    # V[t] will store the value function at period t over the state grid.
    V = [zeros(nK, nM) for t in 1:(T+1)]
    # Policy functions for consumption and portfolio at each period.
    policy_c = [fill(NaN, nK, nM) for t in 1:T]
    policy_portfolio = [Array{Tuple{Float64,Float64}}(undef, nK, nM) for t in 1:T]
    
    # Terminal condition: at t = T+1, assume the policymaker consumes all available resources.
    # For simplicity, we compute production using the default portfolio p.portfolio.
    for i in 1:nK, j in 1:nM
        K_val = K_grid[i]
        M_val = M_grid[j]
        L_T = compute_L(T, p.L0, p.n)
        tech = compute_portfolio_metrics(p.portfolio, T)
        A_T = tech.A
        T_temp = p.α * log(M_val/p.M0) / log(2.0)
        D_val = compute_damage(T_temp, p.θ1, p.θ2; σ=p.σ)
        Y = compute_Y(A_T, K_val, L_T) * D_val
        c_terminal = (1 - p.δ)*K_val + Y
        V[T+1][i, j] = u(c_terminal, p.η)
    end
    
    # Discretize the portfolio decision space.
    # Each decision is (w_Agriculture, w_Manufacturing) with constraint: w_Agriculture + w_Manufacturing ≤ 1.
    d_port = 0.2
    portfolio_choices = Tuple{Float64,Float64}[]
    for w_ag in 0.0:d_port:1.0
        for w_man in 0.0:d_port:(1.0 - w_ag)
            push!(portfolio_choices, (w_ag, w_man))
        end
    end
    
    n_c = 10  # number of consumption grid points (per state/portfolio combination)
    
    # Backward induction: loop from period T down to 1.
    for t in T:-1:1
        println("Solving period ", t)
        V_next = V[t+1]
        # Create an interpolator for the continuation value over (K, M).
        itp = interpolate(V_next, BSpline(Linear()))
        V_next_itp = Interpolations.scale(itp, K_grid, M_grid)
        
        for i in 1:nK, j in 1:nM
            K_val = K_grid[i]
            M_val = M_grid[j]
            L_t = compute_L(t, p.L0, p.n)
            best_val = -Inf
            best_c = NaN
            best_port = (NaN, NaN)
            
            for port in portfolio_choices
                tech = compute_portfolio_metrics(port, t)
                A_t = tech.A
                T_temp = p.α * log(M_val/p.M0) / log(2.0)
                D_val = compute_damage(T_temp, p.θ1, p.θ2; σ=p.σ)
                Y_val = compute_Y(A_t, K_val, L_t) * D_val
                
                # Adjust the consumption upper bound so that next capital (K_new) is at least K_min.
                feasible_c_max = (1 - p.δ)*K_val + Y_val - K_min
                
                # Define a grid for consumption.
                if feasible_c_max > 1e-3
                    c_grid = collect(range(max(1e-3, 0.01*feasible_c_max), feasible_c_max, length=n_c))
                else
                    c_grid = [max(1e-3, 0.01*feasible_c_max)]
                end
                                
                for c in c_grid
                    # Transition to next state based on chosen c and portfolio.
                    K_new, M_new, reward = dp_transition(p, t, K_val, M_val, c, port)
                    
                    # Reject infeasible transitions (outside our grid bounds)
                    if (K_new < K_min) || (K_new > K_max) || (M_new < M_min) || (M_new > M_max) || isnan(K_new) || isnan(M_new)
                        cont_val = -1e6  # heavy penalty for infeasibility
                    else
                        cont_val = V_next_itp(K_new, M_new)
                    end
                    total_val = reward + beta * cont_val
                    if total_val > best_val
                        best_val = total_val
                        best_c = c
                        best_port = port
                    end
                end
            end
            V[t][i,j] = best_val
            policy_c[t][i,j] = best_c
            policy_portfolio[t][i,j] = best_port
        end
    end
    
    return (V=V, policy_c=policy_c, policy_portfolio=policy_portfolio, K_grid=K_grid, M_grid=M_grid, p=p)
end

# -----------------------------------------------------------------------------------
# Example: Run the DP solver and plot the optimal consumption policy at t=1.
# -----------------------------------------------------------------------------------



# Check if solution exists in cache, otherwise compute and save it
solution_file = "dp_solution_new_new_new_new_new.jld2"
if isfile(solution_file)
    dp_solution = load(solution_file, "dp_solution") 
    println("Loaded DP solution from cache")
else
    dp_solution = dynamic_program_solver()
    save(solution_file, "dp_solution", dp_solution)
    println("Computed and saved new DP solution")
end

# Unpack solution
V = dp_solution.V
policy_c = dp_solution.policy_c
policy_portfolio = dp_solution.policy_portfolio
K_grid = dp_solution.K_grid
M_grid = dp_solution.M_grid
p = dp_solution.p

report_results(dp_solution)
