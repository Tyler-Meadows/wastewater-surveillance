## Create probabilities for Figures

using Revise
using ParticleFilter
using CSV, Dates, XLSX
using Optim, DataFrames
using StatsPlots
includet("SIRModel.jl")
## Load data 

Towns = ["Small_City","Rural_Town_1","Rural_Town_2","Rural_Town_3","Rural_Town_4","Rural_Town_5"]
function Measure_virus(measurement::DataFrameRow,part::Particle)
    #ismissing(measurement.concentration_per_liter_corrected) && return part.weight
    @unpack μ,σ,p = part.stats_pars
    V = measurement.flow_rate*measurement.concentration_per_liter_corrected
    l1 =  likelihood(Gamma(rate_and_scale(μ,σ)...),V,part.state[3])
    return l1
end

function probabilities(History::FilterHistory)
    NT = length(History.T)
    N = sum(History.particles[1][1].state)+1
    probs = zeros(Float64,(NT,Int(N),length(History.particles[1][1].state)))
    for t in 1:NT
        for p in History.particles[t]
            for (j,s) in enumerate(Int.(p.state))
            probs[t,s+1,j] += p.weight
            end
        end
    end
    return probs./sum(probs,dims=2)    
end


for Town in Towns
    print(Town)
    town_data = CSV.File("output/$(Town)_town_data.csv") |> DataFrame
    #=
    Mean and of virus shedding data
    estimated using the function discussed in 
        Phan T., Brozac K., Pell  B., Gitter A. Mena K.D., Kuang Y., and Wu F. (2023)
        A simple SEIR-V model to estimate COVID-19 Prevalence and predict SARS-CoV-2
        Transmission using wastewater-based surveillance data. Sci. Total Env. 857, 159326
    The variance is assumed to be on the same order of magnitude as the mean, which
    matches the data collected in 
        Wolfel, ....

    =#
    f(t) = 71.97*t/(16+t^2)
    E_I = 1/8*sum(f.(3.0:0.001:11.0).*0.001) |> (x-> 10^x)
    V_I = E_I
    #- Generate synthetic data -#
    WW_stats_parameters = [E_I, V_I]
    Adjusted = WW_stats_parameters.*(128)/3.78541e6



    subset = filter(x->x.date ≤ Date(2022,05,03),town_data)
    !issorted(subset,:date) && sort!(subset,:date)

    ℒ = []
    pars = (β = 0.3, τ =3.0, γ = 8.0)
    stats_pars = (μ = Adjusted[1] , σ = Adjusted[2], p = 0.8)
    pfilter = Filter(1,[],subset,Measure_virus,SEIR!,init_filter!)
    pars, ℒ =  ParticleFilter.iterated_filtering!(pfilter,
                                    init_filter!,
                                    10000,
                                    100,
                                    pars,
                                    stats_pars,
                                    0.95,
                                    [0.001,0.005,0.005];
                                    Track_Likelihood = true)
    file = open("output/parameter_fits_corrected.csv","a")
    write(file,"$Town $pars \n")
    close(file)


    pfilter= Filter(1,[],subset,Measure_virus,SEIR!,init_filter!)
    init_filter!(pfilter, 50000, pars, stats_pars)
    History = run_filter!(pfilter); 



    probs = probabilities(History)
    infected = probs[:,1:200,3]
    #=
    heatmap(subset.date,0:(size(infected,2)-1),
        infected', color = cgrad([RGBA(1,1,1),RGBA(20/255,66/255,129/255)]),
                colorbar_title = "Probability", grid = false)
    ylims!(0,40)
    @df subset plot!(:date,:cases)
    =#
    ForThibault = DataFrame(probs[:,:,3],:auto)
    ForThibault.date = subset.date
    CSV.write("output/$(Town)_probabilities_corrected.csv",ForThibault)
end
