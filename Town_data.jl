## Sorting the data for each town
# This script takes the data from "case_number_by_county.csv" and 
# "samples.QL.csv" and collates it into the CSV files for each town/city.

using CSV, DataFrames
using Dates
using StatsBase

function LoadData(Town::String,)

end


case_counts = CSV.File("data/case_number_by_county.csv";missingstring=["","NA"]) |> DataFrame


ww_meta_data = CSV.File("data/samples.QL.csv";missingstring="NA") |> DataFrame
rename!(ww_meta_data, :sample_collect_date => :date)
filter!(x->x.pcr_gene_target == "N1",ww_meta_data )
sort!(ww_meta_data,:date)

for Town in ["Small_City","Rural_Town_1","Rural_Town_2","Rural_Town_3","Rural_Town_4","Rural_Town_5"]
    ## Full range of dates
    town_data = DataFrame(date = ww_meta_data.date[1]:Day(1):ww_meta_data.date[end])
    # Filter out non-town locations

    town_data = leftjoin(town_data,filter(x-> x.location == Town, ww_meta_data), on= :date)
    filter!(x-> x.date ∈ town_data.date,case_counts)

    town_data = leftjoin(town_data,case_counts, on = :date)
    town_data[:,"$Town"] = coalesce.(town_data[:,"$Town"],0)
    !issorted(town_data.date) && sort!(town_data, :date)
    transform!(town_data, "$Town" => (x->Rolling_sum(x,10)) => :cases)



    transform!(town_data,:population_served => (x->coalesce.(x,x[findfirst(!ismissing,x)])) => :population_served)

    ## some flow rates are missing, use most recent future measurement


    town_data.flow_rate = replace(town_data.flow_rate, nothing => missing)
    mean_flow = median(skipmissing(town_data.flow_rate))
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
    CSV.write("data/$(Town)_town_data.csv",town_data)
end
