## Before you run this code, be aware that it can take several days on a personal computer.

include("ROC_Curve.jl")
include("Synthetic_data.jl")
print("")

AUC_Frame = DataFrame(:RunID => Float64[], :Day => Int64[], :AUC => Float64[])
for runID in 1:50
    print("\u1b[1A Run $runID of 50")
    Epi_final = 0
    while Epi_final < 30
        sol = solve(SEIR_data!,u0,(0,600),(β = 0.2, τ=3, γ=8,))
        sol = DataFrame(sol)
        Shed!(sol,3)
        sol.population_served .= 2000
        Epi_final = findlast(x->x >0, sol.x3)+5
    end
    filter!(x-> rownumber(x) ≤ Epi_final,sol)
    pfilt = Filter(1,[],sol,MeasurementModel,SEIR!,init_filter!)

    pars = (β = 0.2, τ = 3, γ = 8)
    pfilt.T = 1
    init_filter!(pfilt,
                10000,
                pars,
                (μ = E_I, σ = V_I))



    history = run_filter!(pfilt);                    


    for days in 1:15
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
        push!(AUC_Frame,[runID,days,mean(Roc_data[2,:])])
    end
    
end

@df AUC_Frame boxplot(:Day,:AUC,
                label = nothing,
                xlabel = "Days Forecast",
                ylabel = "AUC")
savefig("figures/AUC_E.pdf")
CSV.write("output/AUC_Data.csv",AUC_Frame,append=true)
