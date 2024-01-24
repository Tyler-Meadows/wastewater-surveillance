## SEIR Model


using Distributions, StatsBase
using Parameters
using DataFrames
using SpecialFunctions
#using StatsPlots


#= Initialization of particles =#
# Create a probability distribution from the likelihoods to sample
function likelihood_distribution(pfilt::Filter,pars)
    dist = Array{Float64}(undef,(100,100))
    items = Array{Tuple}(undef,(100,100))
    measurements = pfilt.Measurements[pfilt.T,:]
    for i in 1:100
        for j in 1:100
            items[i,j] = (i-1,j-1)
            dist[i,j] = pfilt.MeasurementModel(measurements,Particle(1.0,[0,i-1,j-1],pars,pars))*pdf(Poisson(i),j-1)
        end
    end
    dist = dist/sum(dist)
    return items, dist
end

"""
    rate_and_scale(μ::Float64,σ::Float64)
Calculates the rate and scale of a gamma distribution with mean μ and variance σ using the method of moments. 
"""
function rate_and_scale(μ,σ)
        return [μ^2/σ^2,σ^2/μ]
end

"""
    likelihood(dist::Gamma,V,N)
Calculates the likelihood ℒ (N|V).
Due to large numbers, this uses Stirling's approximation to calculate the log likelihood.
"""
function likelihood(dist::Gamma,V,N)
    @unpack α,θ = dist
    V == N == 0 && return 1.0
    N == 0 && return exp(-V/θ)/θ
    V == 0 && return exp(-N/θ)/θ
    (N*α-1)*log(V)-V/θ-N*α*log(θ)-α*N*log(α*N)+α*N+1/2*log(α*N/2π) |> exp
end
function likelihood(dist::Gamma,V::Missing,N)
    1.0
end

function likelihood(dist::Normal,V,N)
    N == 0 && return pdf(Dirac(0.0),V)
    @unpack μ,σ = dist
    1/sqrt(2π*N*σ^2)*exp(-(V-N*μ)^2/(2*N*σ^2))
end


function init_filter!(p::Filter,
        N_particles::Int,
        pars::NamedTuple,
        statisticalparameters::NamedTuple)
        p.particles = Array{Particle}(undef,N_particles) #initialize particle array to avoid races
        like, wv = likelihood_distribution(p,statisticalparameters)
        total = p.Measurements.population_served[1]
        Threads.@threads for i in 1:N_particles
            E,I = sample(like,weights(wv))
            S = Binomial(total-E-I,0.9) |> rand
            p.particles[i] = Particle(1/N_particles,
                            [S,E,I,total-S-E-I],
                            pars,
                            statisticalparameters)
        end
end

## Reinitialize
function init_filter!(p::Filter)
    init_filter!(p,
                length(p.particles),
                average_particle(p).pars,
                p.particles[1].stats_pars)
end

## Dynamic Model(s)
function infect!(du,r)
    du[1] -= r
    du[2] += r
end
function progress!(du,r)
    du[2] -= r
    du[3] += r
end
function recover!(du,r)
    du[end-1] -= r
    du[end] += r
end

function lose_immunity!(du,r)
    du[end] -= r
    du[1] += r
end

function SIR!(du,u,t,p)
    @unpack β,τ,γ = p
    du .= zero(du)
    infect!(du,Poisson(β[1]*u[1]*u[2]/sum(u)) |> rand)
    recover!(du,Binomial(u[2],1.0/γ)|>rand)
    nothing
end

function SEIR!(du,u,t,p)
    @unpack β,τ,γ = p
    du .= zero(du)
    #infect!(du,Binomial(u[1],β*u[3]/sum(u)) |> rand)
    infect!(du,Binomial(u[1],β*u[3]/sum(u)) |> rand)
    progress!(du,Binomial(u[2],1.0/τ) |> rand)
    recover!(du,Binomial(u[3],1.0/γ)|>rand)
    #lose_immunity!(du,Binomial(u[end],1.0/90.0)|>rand)
    nothing
end

# out of place dynamic model
function SEIR(u,t,p)
    @unpack β,τ,γ = p
    du .= zero(u)
    infect!(du,Poisson(u[1],β*u[3]/sum(u)) |> rand)
    progress!(du,Binomial(u[2],1.0/τ) |> rand)
    recover!(du,Binomial(u[3],1.0/γ)|>rand)
    return u .+ du
end

import Statistics.var
function var(p::Filter)
    weights = [part.weight for part in p.particles]
    states = [part.state for part in p.particles]
    μ = average_particle(p)
    for i in 1:length(weights)
        states[i] = weights[i].*(states[i]-μ.state).^2.0
    end
    sum(states,dims=1)[1]
end

function Rolling_average(Vec::Vector,n::Int)
    av = zeros(length(Vec))
    for i in 1:n
         av[i] = mean(Vec[1:i])
    end
    if length(Vec)>n
        for i in (n+1):length(Vec)
            av[i] = mean(Vec[(i-n):i])
        end
    end
    return av
end
function Rolling_sum(Vec::Vector,n::Int)
    av = similar(Vec)
    for i in 1:n
         av[i] = sum(skipmissing(Vec[1:i]))
    end
    if length(Vec)>n
        for i in (n+1):length(Vec)
            if isempty(skipmissing(Vec[(i-n):i]))
                av[i] = missing
            else
               av[i] = sum(skipmissing(Vec[(i-n):i]))
            end
        end
    end
    return av
end

function test_range(mu,var)
    @assert length(mu) == length(var) "Length of mu and var do not match"
    tmp = []
    for (k,v) in enumerate(var)
        push!(tmp,(mu[k]-v[1],v[2]-mu[k]))
    end
    return tmp
end
