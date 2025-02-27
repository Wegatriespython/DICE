using Statistics
using Plots
using Interpolations
using Printf
# Ensure that the necessary functions are available.
# It is assumed these are imported from DICE_DP.jl and TechPortfolio.jl:
#   compute_L, compute_Y, compute_damage, update_K, and compute_portfolio_metrics

# -----------------------------------------------------------------------------
# Helper function: simulate_dp
#
# For DP results, this function simulates a forward trajectory starting from
# a chosen initial state (default: K0 = 223, M0 from parameters).
#
# It uses the optimal policy functions computed on the DP grid to extract
# consumption and portfolio decisions, then computes the evolution of other
# key variables.
# -----------------------------------------------------------------------------
function simulate_dp(dp_sol; K0=223.0, M0=dp_sol.p.M0)
    p = dp_sol.p
    T_plan = p.T_plan

    # Preallocate arrays
    K = zeros(T_plan+1)
    M = zeros(T_plan+1)
    C = zeros(T_plan)
    Y = zeros(T_plan)
    T_arr = zeros(T_plan)
    D = zeros(T_plan)
    E = zeros(T_plan)
    portfolio_dec = Vector{Tuple{Float64,Float64}}(undef, T_plan)

    # Set initial conditions
    K[1] = K0
    M[1] = M0

    for t in 1:T_plan
        # Find the closest indices in the DP state grids.
        i = findmin(abs.(dp_sol.K_grid .- K[t]))[2]
        j = findmin(abs.(dp_sol.M_grid .- M[t]))[2]

        # Retrieve control decisions from the DP solution.
        c_t = dp_sol.policy_c[t][i, j]
        port_t = dp_sol.policy_portfolio[t][i, j]
        portfolio_dec[t] = port_t
        C[t] = c_t

        # Compute production and climate metrics.
        L_t = compute_L(t, p.L0, p.n)
        tech = compute_portfolio_metrics(port_t, t)
        A_t = tech.A
        E_intensity = tech.E

        T_temp = p.α * log(M[t] / p.M0) / log(2.0)
        T_arr[t] = T_temp
        D[t] = compute_damage(T_temp, p.θ1, p.θ2; σ=p.σ)
        Y_prod = compute_Y(A_t, K[t], L_t)
        Y[t] = Y_prod * D[t]
        E[t] = E_intensity * K[t]^0.4

        # Transition to the next state.
        tmpK = update_K(K[t], Y[t], c_t, p.δ)
        # Use the lower bound from the DP grid (first element of dp_sol.K_grid)
        K_lower = first(dp_sol.K_grid)
        K[t+1] = tmpK < K_lower ? K_lower : tmpK
        
        # Carbon cycle update with the same parameters as in the DP solver:
        phi_atm = 0.2
        phi_up  = 0.32
        M[t+1] = M[t] + E[t] / 3.666
    end

    return (K = K, M = M, C = C, Y = Y, T = T_arr, D = D, E = E, portfolio_dec = portfolio_dec, p = p)
end

# -----------------------------------------------------------------------------
# Report key results to terminal.
#
# If the input results include a DP solution (detected via the presence of
# 'policy_c'), then simulate a forward trajectory before reporting.
# -----------------------------------------------------------------------------
function report_results(results, time_points=[1, 20, 50, 100, 200])
    # Check for DP solution: if 'policy_c' exists, simulate the policy trajectory.
    if haskey(results, :policy_c)
        println("Detected DP solution. Simulating forward trajectory using optimal policy functions.")
        results = simulate_dp(results)
    end

    # Filter time points to ensure they're within range.
    valid_points = filter(t -> t <= length(results.Y), time_points)

    println("\n========== DICE MODEL RESULTS ==========")
    println("\nValues at specific years:")

    # Create header with time points.
    header = "Variable"
    for t in valid_points
        header *= "\tYear $t"
    end
    println(header)

    # Local helper function for printing a variable's values.
    function print_variable(name, values)
        line = name
        for t in valid_points
            if t <= length(values)
                line *= "\t$(round(values[t], digits=2))"
            else
                line *= "\tN/A"
            end
        end
        println(line)
    end

    # Economic metrics
    print_variable("GDP (trillion USD)", results.Y)
    print_variable("Consumption (trillion USD)", results.C)
    print_variable("Capital (trillion USD)", results.K[1:end-1])
    print_variable("Per capita consumption (USD1000)", results.C ./ (results.p.L0 * (1 .+ results.p.n).^(1:results.p.T_plan) ./ 1000))

    # Climate metrics
    print_variable("Emissions (GtCO2/yr)", results.E)
    print_variable("Temperature (°C)", results.T)
    print_variable("CO2 (GtC)", results.M[1:end-1])

    # Damage metrics
    print_variable("Damage (% of GDP)", (1 .- results.D) .* 100)

    # Technology portfolio composition (only if portfolio decisions are available)
    if haskey(results, :portfolio_dec)
        print_variable("Agriculture Share", map(x -> x[1], results.portfolio_dec))
        print_variable("Manufacturing Share", map(x -> x[2], results.portfolio_dec))
        print_variable("Services Share", map(x -> 1.0 - (x[1] + x[2]), results.portfolio_dec))
    end

    # Calculate aggregate metrics.
    discount_factors = (1 ./ (1 .+ results.p.ρ)).^(0:results.p.T_plan-1)
    npv_consumption = sum(results.C .* discount_factors)
    total_emissions = sum(results.E)
    max_temp = maximum(results.T)

    println("\nAggregate metrics:")
    println("NPV of consumption: $(round(npv_consumption, digits=2)) trillion USD")
    println("Cumulative emissions: $(round(total_emissions, digits=2)) GtCO2")
    println("Maximum temperature: $(round(max_temp, digits=2)) °C")

    if valid_points[end] < 200
        println("\nNote: Model horizon ($(length(results.Y)) years) is shorter than some requested time points.")
    end

    println("\n==========================================")
end

# -----------------------------------------------------------------------------
# Visualization function remains largely the same.
# -----------------------------------------------------------------------------
function plot_results(results, baseline=nothing)
    # If a DP solution is detected, simulate a forward trajectory.
    if haskey(results, :policy_c)
        results = simulate_dp(results)
    end

    if isa(results, Tuple)
        # Single result case.
        results = [results]
    end

    labels = ["GDP" "Consumption" "Capital" "Emissions" "CO2" "Temperature" "Damage"]
    plt = plot(layout=(4,2), size=(1200,1000), title=labels)

    # Plot each simulation.
    for (i, res) in enumerate(results)
        plot!(plt[1], res.Y, label="Sim $i", lw=1.5, alpha=0.7)
        plot!(plt[2], res.C, lw=1.5, alpha=0.7)
        plot!(plt[3], res.K[1:end-1], lw=1.5, alpha=0.7)
        plot!(plt[4], res.E, lw=1.5, alpha=0.7)
        plot!(plt[5], res.M[1:end-1], lw=1.5, alpha=0.7)
        plot!(plt[6], res.T, lw=1.5, alpha=0.7)
        plot!(plt[7], res.D, lw=1.5, alpha=0.7)
    end

    # Add baseline as thicker line if provided.
    if baseline !== nothing
        plot!(plt[1], baseline.Y, label="Baseline", lw=2.5, color=:black)
        plot!(plt[2], baseline.C, lw=2.5, color=:black)
        plot!(plt[3], baseline.K[1:end-1], lw=2.5, color=:black)
        plot!(plt[4], baseline.E, lw=2.5, color=:black)
        plot!(plt[5], baseline.M[1:end-1], lw=2.5, color=:black)
        plot!(plt[6], baseline.T, lw=2.5, color=:black)
        plot!(plt[7], baseline.D, lw=2.5, color=:black)
    end

    # Add legend only to the first subplot.
    plot!(plt[1], legend=:topleft)

    # Save and display the plot.
    savefig(plt, "dice_results.png")
    display(plt)
    return plt
end