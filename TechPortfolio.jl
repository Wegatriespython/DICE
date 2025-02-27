module TechPortfolio

export compute_portfolio_metrics

# Define sector-specific technology parameters for Agriculture, Manufacturing, and Services
const tech_params = Dict(
    :Agriculture => (A0 = 1.0, g = 0.001, E0 = 1.2, d = 0.0005),
    :Manufacturing => (A0 = 1.2, g = 0.0011, E0 = 1.5, d = 0.0001),
    :Services => (A0 = 0.8, g = 0.009, E0 = 0.5, d = 0.0002)
)

"""
    compute_portfolio_metrics(portfolio::Tuple{Float64, Float64}, t::Int)

Given a portfolio decision as a tuple (w_Agriculture, w_Manufacturing) representing weights for Agriculture and Manufacturing,
the weight for Services is implicitly 1 - (w_Agriculture + w_Manufacturing).

It returns a named tuple with aggregate productivity (A) and aggregate emission intensity (E)
at time t.
"""
function compute_portfolio_metrics(portfolio::Tuple{Float64, Float64}, t::Int)
    w_ag, w_man = portfolio
    w_ser = 1.0 - w_ag - w_man
    if w_ser < 0
        error("Invalid portfolio weights: Sum exceeds 1.0")
    end
    
    # Compute effective productivity at time t for each sector
    A_ag = tech_params[:Agriculture].A0 * (1 + tech_params[:Agriculture].g)^t
    A_man = tech_params[:Manufacturing].A0 * (1 + tech_params[:Manufacturing].g)^t
    A_ser = tech_params[:Services].A0 * (1 + tech_params[:Services].g)^t
    
    A_eff = w_ag * A_ag + w_man * A_man + w_ser * A_ser

    # Compute effective emission intensity at time t for each sector
    E_ag = tech_params[:Agriculture].E0 * (1 - tech_params[:Agriculture].d)^t
    E_man = tech_params[:Manufacturing].E0 * (1 - tech_params[:Manufacturing].d)^t
    E_ser = tech_params[:Services].E0 * (1 - tech_params[:Services].d)^t

    E_eff = w_ag * E_ag + w_man * E_man + w_ser * E_ser

    return (A = A_eff, E = E_eff)
end

end # module TechPortfolio 