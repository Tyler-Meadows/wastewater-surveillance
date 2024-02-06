## Fit through 2021, predict through 2022. 

using Revise
using ParticleFilter
using CSV, Dates, XLSX
using Optim, DataFrames
using StatsPlots
includet("SIRModel.jl")
## Load data 
Towns = ["Small_City","Rural_Town_1","Rural_Town_2","Rural_Town_3","Rural_Town_4","Rural_Town_5"]


for Town in Towns
    town_data = CSV.File("data/$(Town)_town_data.csv") |> DataFrame

    populations = Dict("Rural_Town_3" => 906,
                    "Small_City" => 25850,
                    "Rural_Town_1" => 1044,
                    "Rural_Town_2" => 744,
                    "Rural_Town_4" => 632,
                    "Rural_Town_5" => 291)
    transform!(town_data,:population_served => (x->coalesce.(x,populations[Town])) => :population_served)

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
        E_I = 1/3*sum(f.(0.0:0.001:3.0).*0.001) |> (x-> 10^x)
        V_I = 0.5*E_I
        #- Measurement parameters -#
        WW_stats_parameters = [E_I, V_I]
        Adjusted = WW_stats_parameters.*(128)/3.78541e6

    ## Measurement Model
    function Measure_cases(measurement::DataFrameRow,part::Particle)
        #ismissing(measurement.concentration_per_liter_corrected) && return part.weight
        @unpack μ,σ,p = part.stats_pars
        V = measurement.flow_rate*measurement.concentration_per_liter
        N = measurement.cases
        #l1 =  likelihood(Gamma(rate_and_scale(μ,σ)...),V,part.state[2])
        l2 =  pdf(Normal(part.state[3],sqrt(part.state[3]/4)+0.5),N/p)
        return l2
    end
    function Measure_virus(measurement::DataFrameRow,part::Particle)
        #ismissing(measurement.concentration_per_liter_corrected) && return part.weight
        @unpack μ,σ,p = part.stats_pars
        V = measurement.flow_rate*measurement.concentration_per_liter
        l1 =  likelihood(Gamma(rate_and_scale(μ,σ)...),V,part.state[2])
        return l1
    end

    outbreak_start = Dict("Small_City" => Date(2022,01,04),
                        "Rural_Town_1" => Date(2022,01,06),
                        "Rural_Town_2" => Date(2022,01,01),
                        "Rural_Town_3" => Date(2022,01,14),
                        "Rural_Town_4" => Date(2022,01,05),
                        "Rural_Town_5" => Date(2022,01,12))
    outbreak_peak = Dict("Small_City" => Date(2022,01,18),
                        "Rural_Town_1" => Date(2022,01,09),
                        "Rural_Town_2" => Date(2022,01,25),
                        "Rural_Town_3" => Date(2022,01,18),
                        "Rural_Town_4" => Date(2022,01,11),
                        "Rural_Town_5" => Date(2022,01,27))
    subset = filter(x->x.date ≤ outbreak_peak[Town]+Day(4),town_data)
    !issorted(subset,:date) && sort!(subset,:date)

    ℒ = []
    io = open("output/parameter_fits_2021.csv")
    lines = readlines(io)
    line = lines[findlast(x-> occursin(Town,x),lines)]
    pars = eval(Meta.parse(line[(length(Town)+2):end]))
    close(io)
    stats_pars = (μ = Adjusted[1] , σ = Adjusted[2], p = 0.8)
    #= Iterated filtering 
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
    =#


    # Predictions from January 2022 onwards

    # Set the measurements for January 2022 onwards to missing in order
    # to not implement any filtering.

    town_data.concentration_avg = convert(Vector{Union{Float64,Missing}},town_data.concentration_avg)
    for j in (nrow(subset)+1):nrow(town_data)
        town_data.concentration_per_liter[j] = missing
    end


    pfilter= Filter(1,[],town_data,Measure_virus,SEIR!,init_filter!)
    init_filter!(pfilter, 50000, pars, stats_pars)

    History = run_filter!(pfilter)
    function probabilities(History::FilterHistory)
        NT = length(History.T)
        N = convert(Int,sum(History.particles[1][1].state)+1)
        probs = zeros(Float64,(NT,N,length(History.particles[1][1].state)))
        for t in 1:NT
            for p in History.particles[t]
                for (j,s) in enumerate(p.state)
                
                probs[t,convert(Int,s+1),j] += p.weight
                end
            end
        end
        return probs./sum(probs,dims=2)    
    end

    prob = probabilities(History)



    DF = DataFrame(date = town_data.date)

    function average_history(history::FilterHistory)
        series = Vector{Particle}(undef,length(history.T))
        for j in 1:length(history.T)
            series[j] = average_particle(history.particles[j])
        end
        return series
    end

    mean_history = average_history(History)
    mean_infected = [p.state[3] for p in mean_history]
    DF.mean = mean_infected

    status = Vector{String}(undef,nrow(town_data))
    for j in 1:nrow(town_data)
        if town_data.date[j] ≤ outbreak_peak[Town]
            status[j] = "fitted"
        else
            status[j] = "predicted"
        end
    end
    DF.status = status
    CDF = cumsum(prob[:,:,3],dims=2)
    #CDF = replace(CDF, NaN => missing)
    lower = findfirst.(x-> x > 0.025,eachrow(CDF))
    upper = findfirst.(x->x>0.975,eachrow(CDF)).-1

    DF.upper = upper
    DF.lower = lower
    #=
    @df town_data plot(:date,:cases)
    @df DF plot!(:date,:mean)
    @df DF plot!(:date,:upper, color = 2,alpha = 0.5, label = nothing)
    @df DF plot!(:date,:lower,color = 2, alpha = 0.5, label = nothing)
    ylims!(0,100)
    =#
    CSV.write("output/$(Town)_peak+4.csv",DF)
end

