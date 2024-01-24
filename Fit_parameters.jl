using CSV, DataFrames
using ParticleFilter
using StatsPlots
include("SIRModel.jl")
## Import data
sol = CSV.File("output/Synthetic_data.csv") |> DataFrame

# Parameters used in fitting
f(t) = 71.97*t/(16+t^2)
E_I = 1/3*sum(f.(0.0:0.001:3.0).*0.001) |> (x-> 10^x)
V_I = 0.5*E_I
# Measurement Model
function MeasurementModel(measurement::DataFrameRow,part::Particle)
    @unpack μ,σ = part.stats_pars
    α,θ = rate_and_scale(μ,σ)
    likelihood(Gamma(α,θ),measurement.V,part.state[2])
end

pfilt = Filter(1,[],sol,MeasurementModel,SEIR!,init_filter!)

pars = (β = 0.2, τ = 3, γ = 8)
pfilt.T = 1
init_filter!(pfilt,
                   10000,
                   pars,
                   (μ = E_I, σ = V_I))

history  = run_filter!(pfilt);

function probabilities(History::FilterHistory)
    NT = length(History.T)
    N = sum(History.particles[1][1].state)+1
    probs = zeros(Float64,(NT,N,length(History.particles[1][1].state)))
    for t in 1:NT
        for p in History.particles[t]
            for (j,s) in enumerate(p.state)
            probs[t,s+1,j] += p.weight
            end
        end
    end
    return probs./sum(probs,dims=2)    
end

pro = probabilities(history);

heatmap(1:size(pro,1),
        0:(size(pro,2)-1), 
        sqrt.(pro[1:end,:,3]'),
        color = cgrad([RGBA(1,1,1),RGBA(20/255,66/255,129/255)]),
        colorbar_title = "Probability",
        grid = false)
@df sol[:,:] plot!(:t,:x3,
         label="Recorded cases (Last 10 Days)",
         style = :dash, lw = 3, )
         ylims!(0,maximum(sol.x3)+10)
title!("Synthetic Data")
ylabel!("Active Cases")
xlabel!("Days from start of outbreak")
plot!(legend = :topleft)
savefig("figures/SyntheticFit.png")

