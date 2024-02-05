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
    case_counts = XLSX.readtable("data/case_number_by_county.xlsx","Sheet1") |> DataFrame
    for name in names(case_counts)[2:end]
        rename!(case_counts, name => name[1:end-6])
    end


    ww_meta_data = CSV.File("data/samples.QL.csv") |> DataFrame
    rename!(ww_meta_data, :sample_collect_date => :date)
    filter!(x->x.pcr_gene_target == "N1",ww_meta_data )
    sort!(ww_meta_data,:date)
    unique!(ww_meta_data)
    filter!(x->!ismissing(x.location),ww_meta_data)
    filter!(x->x.location == Town,ww_meta_data)
    transform!(ww_meta_data,:flow_rate => (x-> tryparse.(Float64,x)) => :flow_rate)
    ## Full range of dates
    town_data = DataFrame(date = ww_meta_data.date[1]:Day(1):ww_meta_data.date[end])
    # Filter out non-town locations
    town_data = leftjoin(town_data,ww_meta_data, on= :date)
    filter!(x-> x.date ∈ town_data.date,case_counts)
    select!(case_counts,["date",Town])
    town_data = leftjoin(town_data,case_counts, on = :date)
    town_data[:,"$Town"] = coalesce.(town_data[:,"$Town"],0)
    !issorted(town_data.date) && sort!(town_data, :date)
    transform!(town_data, "$Town" => (x->Rolling_sum(x,10)) => :cases)


    populations = Dict("Troy" => 862,
                    "Moscow" => 24000,
                    "Genesee" => 946,
                    "Potlatch" => 994,
                    "Juliaetta" => 609,
                    "Kendrick" => 369)
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
        E_I = 1/8*sum(f.(3.0:0.001:11.0).*0.001) |> (x-> 10^x)
        V_I = 0.5*E_I
        #- Generate synthetic data -#
        WW_stats_parameters = [E_I, V_I]
        Adjusted = WW_stats_parameters.*(128)/3.78541e6


    ## some flow rates are missing, use most recent future measurement
    ## Kendrick and Troy have no flow_rate measurements at all

    town_data.flow_rate = replace(town_data.flow_rate, nothing => missing)
    mean_flow = mean(skipmissing(town_data.flow_rate))
    town_data.flow_rate = coalesce.(town_data.flow_rate,mean_flow)


    # Sometimes there are more than one measurement per day. Average them

    avg_vec = Union{Missing,Float64}[]
    for row in eachrow(town_data)
        df = filter(x-> row.date - Day(10) ≤ x.date ≤ row.date, town_data)
        avg = mean(skipmissing(df.concentration_per_liter))
        push!(avg_vec, avg)
    end
    town_data.concentration_avg = avg_vec

    town_data_grouped = groupby(town_data,:date)
    town_data = combine(town_data_grouped,
                        [:flow_rate,
                        :concentration_per_liter,
                        :population_served,
                        :cases,:concentration_avg] .=> mean .=> [:flow_rate,:concentration_per_liter,:population_served,:cases,:concentration_avg])

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

    outbreak_start = Dict("Moscow" => Date(2022,01,04),
                        "Genesee" => Date(2022,01,06),
                        "Potlatch" => Date(2022,01,01),
                        "Troy" => Date(2022,01,14),
                        "Juliaetta" => Date(2022,01,05),
                        "Kendrick" => Date(2022,01,12))
    outbreak_peak = Dict("Moscow" => Date(2022,01,18),
                        "Genesee" => Date(2022,01,09),
                        "Potlatch" => Date(2022,01,25),
                        "Troy" => Date(2022,01,18),
                        "Juliaetta" => Date(2022,01,11),
                        "Kendrick" => Date(2022,01,27))
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

