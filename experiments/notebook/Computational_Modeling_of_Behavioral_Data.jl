# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Julia 1.4.1
#     language: julia
#     name: julia-1.4
# ---

# # Computational Modeling of Behavioral Data by Prof. Kentaro Katahira
#
# ## Rescorla-Wagner model

# +
using Plots
using Interact
using Random

"""
Nₜ: number of trials
α: learning rate
Pᵣ: probability of getting reward
"""

@manipulate for Nₜ = 0:1:500, α = 0:0.05:1, Pᵣ = 0:0.05:1

    rng = MersenneTwister(1234) #create a seed for random numbers

    𝐕 = zeros(Nₜ) #strengths of association as Nₜ-length vector
    𝐑 = rand(rng, Nₜ) .< Pᵣ # presence of reinforcement (1 or 0) as Nₜ-length vector

    for t in 1: Nₜ-1

        𝐕[t+1] = 𝐕[t] + α *(𝐑[t]-𝐕[t])
    end

    plot(𝐕, label= string("a ", α))
    plot!([(i, Pᵣ) for i in 1:1:Nₜ], label="expected value of r: " * string(Pᵣ))
    xlabel!("number of trials")
    ylabel!("strength of association")
    ylims!((0, 1))
    title!("Rescorla-Wagner model")
end
# -

# ## Q-learning simulation
#
#
# ### softmax function

# +
function softmax(β, Δq)
    return 1 / (1+ exp(-β * (Δq)))
end

@manipulate for β in 0:0.05:5
    plot([(Δq, softmax(β, Δq)) for Δq in -4:0.1:4], m=:o, label=string("beta ", β))
    xlabel!("difference in Q")
    ylabel!("probability")
    ylims!((0, 1))
    title!("Softmax Function")
end
# -

# ### interactive plot of Q-learning model

# +
"""
Nₜ: number of trials
α: learning rate
β: inverse temperature
Pᵣ: probability of getting reward in A
"""

@manipulate for Nₜ in 0:5:200, α in 0:0.05:1, β in 0:0.25:5, Pᵣ in 0:0.05:1

    rng = MersenneTwister(1234)

    𝐐 = zeros((2, Nₜ)) #initial value of Q in 2 by Nₜ matrix
    𝐜 = zeros(Int, Nₜ) #initial choice in each Nₜ trial
    𝐫 = zeros(Nₜ) # 0 (no reward) or 1 (reward) in each Nₜ trial
    Pₐ = zeros(Nₜ) # probability of choosing A in each trial
    P = (Pᵣ, 1-Pᵣ)

    for t in 1:Nₜ-1
        Pₐ = softmax(β, 𝐐[1, t] - 𝐐[2, t])

        if rand(rng) < Pₐ
            𝐜[t] = 1 #choose A
            𝐫[t] = Int(rand(rng) < P[1])
        else
            𝐜[t] = 2 #choose B
            𝐫[t] = Int(rand(rng) < P[2])
        end

        𝐐[𝐜[t], t+1] = 𝐐[𝐜[t], t] + α * (𝐫[t] - 𝐐[𝐜[t], t])
        𝐐[3 - 𝐜[t], t+1] = 𝐐[3 - 𝐜[t], t] # retain value of unpicked choice
    end

    plot(𝐐[1, :], label="Qt(A)", color="orange")
    plot!([(i, P[1]) for i in 1:1:Nₜ], label="expected value of reward for A:" * string(P[1]), color="darkorange")
    plot!(𝐐[2, :], label="Qt(B)", color="skyblue")
    plot!([(i, P[2]) for i in 1:1:Nₜ], label="expected value of reward for B:" * string(P[2]), color="darkblue")
    xlabel!("number of trials")
    ylabel!("Q (value of behavior?)")
    ylims!((0, 1))
    title!("Q-learning model")
end
# -

# ## Parameter Estimation
#
# ### Optimization with Optim package

# +
"""
This function returns a vector of choices and a vector of rewards, both of which will be used for parameter estimation
"""

function generate_qlearning_data(Nₜ, α, β, Pᵣ)

    𝐐 = zeros((2, Nₜ)) #initial value of Q in 2 by Nₜ matrix
    𝐜 = zeros(Int, Nₜ) #initial choice in each Nₜ trial
    𝐫 = zeros(Nₜ) # 0 (no reward) or 1 (reward) in each Nₜ trial
    Pₐ = zeros(Nₜ) # probability of choosing A in each trial
    P = (Pᵣ, 1-Pᵣ)

    for t in 1:Nₜ-1
        Pₐ = softmax(β, 𝐐[1, t] - 𝐐[2, t])

        if rand() < Pₐ
            𝐜[t] = 1 #choose A
            𝐫[t] = (rand(Float64) < P[1])
        else
            𝐜[t] = 2 #choose B
            𝐫[t] = Int(rand(Float64) < P[2])
        end

        𝐐[𝐜[t], t+1] = 𝐐[𝐜[t], t] + α * (𝐫[t] - 𝐐[𝐜[t], t])
        𝐐[3 - 𝐜[t], t+1] = 𝐐[3 - 𝐜[t], t] # retain value of unpicked choice
    end

    return 𝐜, 𝐫
end

# +
"""
init_values: [α, β]
α: learning rate
β: inverse temperature
𝐜: vector of choices in each Nₜ trial in 1(A) or 2(B)
𝐫: 0 (no reward) or 1 (reward) in each Nₜ trial

"""

function func_qlearning(init_values, 𝐜, 𝐫) #needed for parameters to be passed as list for Optim package

    Nₜ = length(𝐜)
    Pₐ = zeros(Nₜ) #probabilities of selecting A
    𝐐 = zeros((2, Nₜ))
    logl = 0 #initial value of log likelihood

    for t in 1:Nₜ - 1
        Pₐ = softmax(init_values[2], 𝐐[1, t] - 𝐐[2, t])
        logl += (𝐜[t] == 1) * log(Pₐ) + (𝐜[t] == 2) * log(1 - Pₐ)
        𝐐[𝐜[t], t + 1] = 𝐐[𝐜[t], t] + init_values[1] * (𝐫[t] - 𝐐[𝐜[t], t])
        𝐐[3 - 𝐜[t], t + 1] =  𝐐[3 - 𝐜[t], t]
    end

    return (negll = -logl, 𝐐 = 𝐐, Pₐ = Pₐ);
end

# +
using Optim

@manipulate for Nₜ in 0:5:200, α in 0:0.05:1, β in 0:0.25:5, Pᵣ in 0:0.05:1
    𝐜, 𝐫 = generate_qlearning_data(Nₜ, α, β, Pᵣ)

    func_qlearning_opt(init_values) = func_qlearning(init_values, 𝐜, 𝐫).negll

    initial_values = rand(2)
    lower = [0.0, 0.0]
    upper = [1.0, 5.0]
    inner_optimizer = GradientDescent()
    results = optimize(func_qlearning_opt, lower, upper, initial_values, Fminbox(inner_optimizer));
end
# -

# #### optimization with BlackBoxOptim package

# +
using BlackBoxOptim

@manipulate for Nₜ in 0:5:200, α in 0:0.05:1, β in 0:0.25:5, Pᵣ in 0:0.05:1
    𝐜, 𝐫 = generate_qlearning_data(Nₜ, α, β, Pᵣ)

    func_qlearning_opt(init_values) = func_qlearning(init_values, 𝐜, 𝐫).negll

    results = bboptimize(func_qlearning_opt; SearchRange = [(0.0, 1.0), (0.0, 5.0)], NumDimensions = 2);
    best_candidate(results)
end
# -

# We can also compare performances when using different optimizers.

𝐜, 𝐫 = generate_qlearning_data(100, 0.3, 1.2, 0.5) #dammy data
func_qlearning_opt(init_values) = func_qlearning([0.3, 0.4], 𝐜, 𝐫).negll
compare_optimizers(func_qlearning_opt; SearchRange = [(0.0, 1.0), (0.0, 5.0)], NumDimensions = 2);

# #### optimization with JuMP and Ipopt packages

# +
#The following code block generates error. How can I fix it?

using JuMP, Ipopt, ForwardDiff

𝐜, 𝐫 = generate_qlearning_data(50, 0.6, 0.7, 0.5)

func_qlearning_JuMP(α, β) = func_qlearning((α, β), 𝐜, 𝐫).negll #JuMP needs separate variables, not a list

m = Model(Ipopt.Optimizer)
register(m, :func_qlearning_JuMP, 2, func_qlearning_JuMP, autodiff=true)

@variable(m, 0.0 <= x <= 1.0, start=rand())
@variable(m, 0.0 <= y <= 5.0, start=5*rand())
@NLobjective(m, Min, func_qlearning_JuMP(x, y))
@show optimize!(m)
println("α = ", value(x), " β = ", value(y))
# -
