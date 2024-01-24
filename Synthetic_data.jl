using Revise
using ParticleFilter
using Parameters, DataFrames
using StatsPlots, Random, Distributions
using CSV
includet("SIRModel.jl")
Plots.theme(:wong)

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
#- Generate synthetic data -#

u0 =[1998,3,0,0]

mutable struct Solution
    t::Vector{Any}
    u::Vector{Any}
end

function SEIR_data!(du,u,t,p)
    @unpack β,τ,γ = p
    du.= 0.0
    infect!(du,Binomial(u[1],β*u[3]/sum(u)) |> rand)
    progress!(du,Binomial(u[2],1.0/τ) |> rand)
    recover!(du,Binomial(u[3],1.0/γ) |> rand)
    nothing
end


function solve(fun,u0,tspan,p) where T <: Function
    sol = Solution([tspan[1]],[u0])
    du = zero(u0)
    for t in (tspan[1]+1):(tspan[2])
        fun(du,sol.u[end],t,p)
        push!(sol.t,t)
        push!(sol.u,sol.u[end].+du)
    end
    return sol
end

import DataFrames.DataFrame
function DataFrame(sol::Solution)
    Df = DataFrame(hcat(sol.u...)',:auto)
    Df = hcat(DataFrame(t=sol.t),Df)
end

function Shed!(Df::DataFrame,freq = 1)
    Df.V = Vector{Union{Float64,Missing}}(undef,nrow(Df))
    for row in eachrow(Df)
        if row.x2 ≥ 0
            if ((rownumber(row)-1) % freq == 0)
                row.V = 0
                for n in 1:row.x2
                 row.V += rand(Gamma(rate_and_scale(E_I,V_I)...));
                end
            else
                row.V = missing
            end
        end 
    end
end


sol = solve(SEIR_data!,u0,(0,600),(β = 0.2, τ=3, γ=8,))
sol = DataFrame(sol)
Shed!(sol,3)
sol.population_served .= 2000
Epi_final = findlast(x->x >0, sol.x3)+5
filter!(x-> rownumber(x) ≤ Epi_final,sol)

plot(sol.t,sol.x3, color = 3, lw=2, legend=nothing, right_margin = 30Plots.mm, ylabel = "Active Cases",)

scatter(sol.t,sol.V)

CSV.write("output/Synthetic_data.csv",sol)
