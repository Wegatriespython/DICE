using Distributions, Plots, Optim

include("Reporting.jl")
include("TechPortfolio.jl")  # New module for technology portfolio
using .TechPortfolio

# Define model parameters
Base.@kwdef struct Params
    # Economic parameters
    ρ = 0.03                # Time preference rate
    η = 1.5                 # Elasticity of marginal utility
    T_plan = 50             # Planning horizon (years)
    δ = 0.1                 # Capital depreciation rate
    A0 = 5.115              # (Deprecated) Global productivity (for legacy code)
    g = 0.015               # (Deprecated) Global productivity growth
    L0 = 7403               # Initial labor (million workers)
    n = 0.001               # Labor growth rate
    
    # Climate parameters
    E0 = 35.0               # (Deprecated) Initial emissions, now replaced by tech portfolio outcome
    M0 = 851.0              # Initial atmospheric CO2 (GtC)
    ϕ = 0.5                 # Carbon retention rate
    α = 3.0                 # Climate sensitivity (°C per doubling CO2)
    
    # Damage parameters (deterministic: σ = 0)
    θ1 = 0.00236            # Linear damage coefficient
    θ2 = 0.0000035          # Quadratic damage coefficient
    σ = 0.0                 # Damage volatility (zero for deterministic)
    
    # Technology portfolio decision
    # Tuple: (w_Agriculture, w_Manufacturing) with Services weight = 1 - (w_Agriculture + w_Manufacturing)
    portfolio = (0.33, 0.33)
end

# CRRA utility function
u(c, η) = c^(1-η)/(1-η)

# Compute labor force at time t
compute_L(t, L0, n) = L0 * (1 + n)^t

# Compute productivity at time t
compute_A(t, A0, g) = A0 * (1 + g)^t

# Compute Cobb-Douglas production
compute_Y(A, K, L) = A * K^0.3 * L^0.7

# Compute climate damage factor (deterministic: σ = 0)
compute_damage(T, θ1, θ2, σ=0.0) = 1 - (θ1*T + θ2*T^2)

# Update capital stock
update_K(K_prev, Y, C, δ) = (1 - δ)*K_prev + Y - C

function social_planner(; kwargs...)
    p = Params(; kwargs...)
    T = p.T_plan

    # Allocate arrays. Note that K, M, and Λ have T_plan+1 elements.
    K     = zeros(T+1)
    M     = zeros(T+1)
    Λ     = zeros(T+1)
    C     = zeros(T)
    Y     = zeros(T)
    T_arr = zeros(T)
    D     = zeros(T)
    E_arr = zeros(T)
    
    # Initial conditions
    K[1] = 223.0             # Initial capital stock
    M[1] = p.M0              # Initial atmospheric CO2
    # Λ[1] will be determined by the shooting method

    # Add more realistic carbon cycle parameters
    phi_atm = 0.2    # Fraction remaining in atmosphere
    phi_up = 0.32    # Fraction going to upper ocean
    
    function objective(λ0)
        # Reset initial conditions for each evaluation
        K[1] = 223.0
        M[1] = p.M0
        Λ[1] = λ0[1]
        
        for t in 1:T
            # Early check to avoid domain errors
            if K[t] <= 0 || Λ[t] <= 0
                return 1e10
            end
            
            # Use the technology portfolio decision to compute effective metrics at time t
            tech = compute_portfolio_metrics(p.portfolio, t)
            
            # Economic fundamentals at time t
            L_t = compute_L(t, p.L0, p.n)
            A_t = tech.A  # Replace legacy production function input with portfolio-based A
            
            # Compute temperature based on current CO2 stock
            T_temp = p.α * log(M[t]/p.M0) / log(2.0)
            T_arr[t] = T_temp
            D[t] = compute_damage(T_temp, p.θ1, p.θ2, p.σ)
            
            # Determine consumption from the shadow price via the Euler relation
            C[t] = Λ[t]^(-1/p.η)
            
            # Compute production and adjust for damages
            Y_prod = compute_Y(A_t, K[t], L_t)
            Y[t] = Y_prod * D[t]
            
            # Update the capital stock
            K[t+1] = update_K(K[t], Y[t], C[t], p.δ)
            
            # Update emissions using portfolio-derived emission intensity (cumulative update):
            E_intensity = tech.E
            E_arr[t] = E_intensity * K[t]^0.4
            if t < T
                M[t+1] = M[t] + E_arr[t] / 3.666
            end
            
            # Update the shadow price using the Euler equation
            if t < T
                MPK = 0.3 * A_t * D[t] * K[t]^(0.3-1) * L_t^0.7
                Λ[t+1] = (1 + p.ρ) * Λ[t] / (1 - p.δ + MPK)
            end
        end
        
        # Terminal condition: the final shadow price should equal the marginal utility of terminal consumption
        terminal_error = abs(Λ[T+1] - C[T]^(-p.η))
        penalty = terminal_error
        
        # Impose additional penalties if state or control variables become nonpositive
        if any(K .<= 0) || any(C .<= 0) || any(Y .<= 0)
            penalty += 1000.0
        end
        
        return penalty
    end

    # Solve for the initial shadow price λ₀ using the shooting method
    res = optimize(objective, [0.5], NelderMead(), Optim.Options(iterations=500))
    
    # Run the final integration with the optimized initial λ₀
    Λ[1] = res.minimizer[1]
    objective(res.minimizer)
    
    return (C=C, K=K, Y=Y, M=M, T=T_arr, Λ=Λ, E=E_arr, D=D, p=p)
end

# Run baseline simulation
result_baseline = social_planner()

# Report baseline results to terminal
time_points = [10, 20, 30, 40, 50]
report_results(result_baseline, time_points)
