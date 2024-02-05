## Create probabilities for Figures

using Revise
using ParticleFilter
using CSV, Dates, XLSX
using Optim, DataFrames
using StatsPlots
includet("SIRModel.jl")
## Load data 

Towns = ["Genesee","Juliaetta","Kendrick","Troy","Potlatch"]
for Town in Towns
    print(Town)
    case_counts = XLSX.readtable("data/case_number_by_county.xlsx","Sheet1") |> DataFrame
    for name in names(case_counts)[2:end]
     rename!(case_counts, name => name[1:end-6])
    end
    ww_meta_data = CSV.File("data/samples.QL.csv") |> DataFrame
    rename!(ww_meta_data, :sample_collect_date => :date)
    filter!(x->x.pcr_gene_target == "N1",ww_meta_data )
    sort!(ww_meta_data,:date)
    unique!(ww_meta_data)
    ## Full range of dates
    town_data = DataFrame(date = ww_meta_data.date[1]:Day(1):ww_meta_data.date[end])
    # Filter out non-town locations
    town_data = leftjoin(town_data,ww_meta_data, on= :date)
    filter!(x->!ismissing(x.location),town_data)
    filter!(x->x.location == Town,town_data)
    filter!(x-> x.date ∈ town_data.date,case_counts)
    select!(case_counts,["date",Town])
    town_data = leftjoin(town_data,case_counts, on = :date)
    town_data[:,"$Town"] = coalesce.(town_data[:,"$Town"],0)
    !issorted(town_data.date) && sort!(town_data, :date)
    transform!(town_data, "$Town" => (x->Rolling_sum(x,8)) => :cases)

    populations = Dict("Troy" => 906,
                    "Moscow" => 25850,
                    "Genesee" => 1044,
                    "Potlatch" => 744,
                    "Juliaetta" => 632,
                    "Kendrick" => 291)
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
    V_I = E_I
    #- Generate synthetic data -#
    WW_stats_parameters = [E_I, V_I]
    Adjusted = WW_stats_parameters.*(128)/3.78541e6

    ## some flow rates are missing, use most recent future measurement
    ## Kendrick and Troy have no flow_rate measurements at all

    town_data.flow_rate = tryparse.(Float64,town_data.flow_rate)
    town_data.flow_rate = replace(town_data.flow_rate, nothing => missing)
    mean_flow = mean(skipmissing(town_data.flow_rate))
    town_data.flow_rate = coalesce.(town_data.flow_rate,mean_flow)

    avg_vec = Union{Missing,Float64}[]
    for row in eachrow(town_data)
        df = filter(x-> row.date - Day(10) ≤ x.date ≤ row.date, town_data)
        avg = mean(skipmissing(df.concentration_per_liter_corrected))
        push!(avg_vec, avg)
    end
    town_data.concentration_avg = avg_vec

    town_data_grouped = groupby(town_data,:date)
    town_data = combine(town_data_grouped,
                        [:flow_rate,
                        :concentration_per_liter_corrected,
                        :population_served,
                        :cases,:concentration_avg] .=> mean .=> [:flow_rate,:concentration_per_liter_corrected,:population_served,:cases,:concentration_avg])


    ## Measurement Model
    function Measure_cases(measurement::DataFrameRow,part::Particle)
        #ismissing(measurement.concentration_per_liter_corrected) && return part.weight
        @unpack μ,σ,p = part.stats_pars
        V = measurement.flow_rate*measurement.concentration_per_liter_corrected
        N = measurement.cases
        #l1 =  likelihood(Gamma(rate_and_scale(μ,σ)...),V,part.state[2])
        l2 =  pdf(Normal(part.state[3],sqrt(part.state[3]/4)+0.5),N/p)
        return l2
    end
    function Measure_virus(measurement::DataFrameRow,part::Particle)
        #ismissing(measurement.concentration_per_liter_corrected) && return part.weight
        @unpack μ,σ,p = part.stats_pars
        V = measurement.flow_rate*measurement.concentration_per_liter_corrected
        l1 =  likelihood(Gamma(rate_and_scale(μ,σ)...),V,part.state[3])
        return l1
    end


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

df = CSV.read("output/Genesee_probabilities_corrected.csv",DataFrame)
w = Matrix(df[:,1:end-1])
t = df[:,end]
heatmap(t,axes(w,2),sqrt.(w)',
    color = cgrad([RGBA(1,1,1),RGBA(20/255,66/255,129/255)]),
    colorbar_title = "Probability",
    grid = false )
ylims!(0,50)
