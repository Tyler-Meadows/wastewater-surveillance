## Create an ROC curve from synthetic data
using CSV, DataFrames
using ParticleFilter
using Parameters
using Plots
include("SIRModel.jl")

#Import Data
sol = CSV.File("output/Synthetic_data.csv") |> DataFrame


# Measurement Model
function MeasurementModel(measurement::DataFrameRow,part::Particle)
    @unpack μ,σ = part.stats_pars
    α,θ = rate_and_scale(μ,σ)
    likelihood(Gamma(α,θ),measurement.V,part.state[2])
end
#Create Particle Filter History
pfilt = Filter(1,[],sol,MeasurementModel,SEIR!,init_filter!)
pars = (β = 0.2, τ = 3, γ = 8)
pfilt.T = 1
init_filter!(pfilt,
                   10000,
                   pars,
                   (μ = E_I, σ = V_I))

history  = run_filter!(pfilt);

## Propogate all particles a few days
function propogate_sample!(model::Function,sample::Particle)
    du = zero(sample.state)
    model(du,sample.state,~,sample.pars)
    sample.state += du
end
function projected_history(history::FilterHistory,days::Int)
    new_history = deepcopy(history)
    for T in new_history.T
        for j in new_history[T][2]
            for k in 1:days
             propogate_sample!(SEIR!,j)
            end
        end
    end
    return new_history
end


function projected_changes(history::FilterHistory,days::Int;Trange = nothing)
    new_history = deepcopy(history)
    if isnothing(Trange)
        Trange = eachindex(new_history.T)
    end
    
    changes = zeros(Int,Trange,eachindex(history.particles[1]),eachindex(history.particles[1][1].state))
    for T in Trange
        for (j,part) in enumerate(new_history.particles[T])
            for k in 1:days
             propogate_sample!(SEIR!,part)
            end
            changes[T,j,:] .= part.state .- history.particles[T][j].state
        end
    end
    range = Base.minimum(changes):Base.maximum(changes)
    chngs = zeros(Int,Trange,range,size(changes,3))
    for l in axes(changes,3)
        for j in axes(changes,1)
            for k in (range)
                chngs[j,k,l] = count(.==(k),changes[j,:,l])
            end
        end
    end
    return chngs./length(new_history.particles[1]), Trange,range
end

## Count Positives, False Positives, True Negatives, False Negatives
@with_kw mutable struct DataStream
    TP::Int64 = 0
    TN::Int64 = 0
    FP::Int64 = 0
    FN::Int64 = 0
end

plot()
for days in 1:2:3
    changes_predicted,T,range  = projected_changes(history,days)

    prob_increase = sum(changes_predicted[:,0:end,3],dims=2)
    ## Determine the actual changes
    changes_real = zeros(Int,length(T)-days)
    for (j,chng) in enumerate(changes_real)
        changes_real[j] = sol.x3[j+days]-sol.x3[j]
    end
    changes_real = sign.(changes_real)
    count(changes_real .≥ 0)

    Roc_data = []
    for α in 0:0.001:1.0
        Rocs = DataStream()
        for j in eachindex(changes_real)
            (changes_real[j]!==-1)&(prob_increase[j]> α)&& (Rocs.TP += 1)
            (changes_real[j]!==-1)&(prob_increase[j]≤ α)&& (Rocs.FN += 1)
            (changes_real[j]==-1)&(prob_increase[j]> α)&&(Rocs.FP += 1)
            (changes_real[j]==-1)&(prob_increase[j]≤ α)&&(Rocs.TN += 1)
        end
        push!(Roc_data,[1.0-α,Rocs.TP/(Rocs.TP+Rocs.FN)])
    end
    
    Roc_data = hcat(Roc_data...)
    plot!(Roc_data[1,:],Roc_data[2,:], label = "$days days")
end
plot!(xlabel = "False Increases", ylabel = "True Increases")
savefig("figures/Sythetic_ROC.png")
