{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Computational Modeling of Behavioral Data by Prof. Kentaro Katahira\n",
    "\n",
    "## Rescorla-Wagner model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "using Plots\n",
    "using Interact\n",
    "\n",
    "\"\"\"\n",
    "Nₜ: number of trials\n",
    "α: learning rate\n",
    "Pᵣ: probability of getting reward\n",
    "\"\"\"\n",
    "\n",
    "@manipulate for Nₜ = 0:1:500, α = 0:0.05:1, Pᵣ = 0:0.05:1\n",
    "\n",
    "    𝐕 = zeros(Nₜ) #strengths of association as Nₜ-length vector\n",
    "    𝐑 = rand(Nₜ) .< Pᵣ # presence of reinforcement (1 or 0) as Nₜ-length vector\n",
    "\n",
    "    for t in 1: Nₜ-1\n",
    "\n",
    "        𝐕[t+1] = 𝐕[t] + α *(𝐑[t]-𝐕[t])\n",
    "    end\n",
    "\n",
    "    plot(𝐕, label= string(\"a \", α))\n",
    "    plot!([(i, Pᵣ) for i in 1:1:Nₜ], label=\"expected value of r: \" * string(Pᵣ))\n",
    "    xlabel!(\"number of trials\")\n",
    "    ylabel!(\"strength of association\")\n",
    "    ylims!((0, 1))\n",
    "    title!(\"Rescorla-Wagner model\")\n",
    "end"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Q-learning simulation\n",
    "\n",
    "\n",
    "### softmax function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "function softmax(β, Δq)\n",
    "    return 1 / (1+ exp(-β * (Δq)))\n",
    "end\n",
    "\n",
    "@manipulate for β in 0:0.05:5\n",
    "    plot([(Δq, softmax(β, Δq)) for Δq in -4:0.1:4], m=:o, label=string(\"beta \", β))\n",
    "    xlabel!(\"difference in Q\")\n",
    "    ylabel!(\"probability\")\n",
    "    ylims!((0, 1))\n",
    "    title!(\"Softmax Function\")\n",
    "end"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### interactive plot of Q-learning model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "Nₜ: number of trials\n",
    "α: learning rate\n",
    "β: inverse temperature\n",
    "Pᵣ: probability of getting reward in A\n",
    "\"\"\"\n",
    "\n",
    "@manipulate for Nₜ in 0:5:200, α in 0:0.05:1, β in 0:0.25:5, Pᵣ in 0:0.05:1\n",
    "\n",
    "    𝐐 = zeros((2, Nₜ)) #initial value of Q in 2 by Nₜ matrix\n",
    "    𝐜 = zeros(Int, Nₜ) #initial choice in each Nₜ trial\n",
    "    𝐫 = zeros(Nₜ) # 0 (no reward) or 1 (reward) in each Nₜ trial\n",
    "    Pₐ = zeros(Nₜ) # probability of choosing A in each trial\n",
    "    P = (Pᵣ, 1-Pᵣ)\n",
    "\n",
    "    for t in 1:Nₜ-1\n",
    "        Pₐ = softmax(β, 𝐐[1, t] - 𝐐[2, t])\n",
    "\n",
    "        if rand() < Pₐ\n",
    "            𝐜[t] = 1 #choose A\n",
    "            𝐫[t] = Int(rand(Float64) < P[1])\n",
    "        else\n",
    "            𝐜[t] = 2 #choose B\n",
    "            𝐫[t] = Int(rand(Float64) < P[2])\n",
    "        end\n",
    "\n",
    "        𝐐[𝐜[t], t+1] = 𝐐[𝐜[t], t] + α * (𝐫[t] - 𝐐[𝐜[t], t])\n",
    "        𝐐[3 - 𝐜[t], t+1] = 𝐐[3 - 𝐜[t], t] # retain value of unpicked choice\n",
    "    end\n",
    "\n",
    "    plot(𝐐[1, :], label=\"Qt(A)\", color=\"orange\")\n",
    "    plot!([(i, P[1]) for i in 1:1:Nₜ], label=\"expected value of reward for A:\" * string(P[1]), color=\"darkorange\")\n",
    "    plot!(𝐐[2, :], label=\"Qt(B)\", color=\"skyblue\")\n",
    "    plot!([(i, P[2]) for i in 1:1:Nₜ], label=\"expected value of reward for B:\" * string(P[2]), color=\"darkblue\")\n",
    "    xlabel!(\"number of trials\")\n",
    "    ylabel!(\"Q (value of behavior?)\")\n",
    "    ylims!((0, 1))\n",
    "    title!(\"Q-learning model\")\n",
    "end"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Parameter Estimation\n",
    "\n",
    "### Optimization with Optim package"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "This function returns a vector of choices and a vector of rewards, both of which will be used for parameter estimation\n",
    "\"\"\"\n",
    "\n",
    "function generate_qlearning_data(Nₜ, α, β, Pᵣ)\n",
    "\n",
    "    𝐐 = zeros((2, Nₜ)) #initial value of Q in 2 by Nₜ matrix\n",
    "    𝐜 = zeros(Int, Nₜ) #initial choice in each Nₜ trial\n",
    "    𝐫 = zeros(Nₜ) # 0 (no reward) or 1 (reward) in each Nₜ trial\n",
    "    Pₐ = zeros(Nₜ) # probability of choosing A in each trial\n",
    "    P = (Pᵣ, 1-Pᵣ)\n",
    "\n",
    "    for t in 1:Nₜ-1\n",
    "        Pₐ = softmax(β, 𝐐[1, t] - 𝐐[2, t])\n",
    "\n",
    "        if rand() < Pₐ\n",
    "            𝐜[t] = 1 #choose A\n",
    "            𝐫[t] = (rand(Float64) < P[1])\n",
    "        else\n",
    "            𝐜[t] = 2 #choose B\n",
    "            𝐫[t] = Int(rand(Float64) < P[2])\n",
    "        end\n",
    "\n",
    "        𝐐[𝐜[t], t+1] = 𝐐[𝐜[t], t] + α * (𝐫[t] - 𝐐[𝐜[t], t])\n",
    "        𝐐[3 - 𝐜[t], t+1] = 𝐐[3 - 𝐜[t], t] # retain value of unpicked choice\n",
    "    end\n",
    "\n",
    "    return 𝐜, 𝐫\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "init_values: [α, β]\n",
    "α: learning rate\n",
    "β: inverse temperature\n",
    "𝐜: vector of choices in each Nₜ trial in 1(A) or 2(B)\n",
    "𝐫: 0 (no reward) or 1 (reward) in each Nₜ trial\n",
    "\n",
    "\"\"\"\n",
    "\n",
    "function func_qlearning(init_values, 𝐜, 𝐫) #needed for parameters to be passed as list for Optim package\n",
    "\n",
    "    Nₜ = length(𝐜)\n",
    "    Pₐ = zeros(Nₜ) #probabilities of selecting A\n",
    "    𝐐 = zeros((2, Nₜ))\n",
    "    logl = 0 #initial value of log likelihood\n",
    "\n",
    "    for t in 1:Nₜ - 1\n",
    "        Pₐ = softmax(init_values[2], 𝐐[1, t] - 𝐐[2, t])\n",
    "        logl += (𝐜[t] == 1) * log(Pₐ) + (𝐜[t] == 2) * log(1 - Pₐ)\n",
    "        𝐐[𝐜[t], t + 1] = 𝐐[𝐜[t], t] + init_values[1] * (𝐫[t] - 𝐐[𝐜[t], t])\n",
    "        𝐐[3 - 𝐜[t], t + 1] =  𝐐[3 - 𝐜[t], t]\n",
    "    end\n",
    "\n",
    "    return (negll = -logl, 𝐐 = 𝐐, Pₐ = Pₐ);\n",
    "end"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "using Optim\n",
    "\n",
    "@manipulate for Nₜ in 0:5:200, α in 0:0.05:1, β in 0:0.25:5, Pᵣ in 0:0.05:1\n",
    "    𝐜, 𝐫 = generate_qlearning_data(Nₜ, α, β, Pᵣ)\n",
    "\n",
    "    func_qlearning_opt(init_values) = func_qlearning(init_values, 𝐜, 𝐫).negll\n",
    "\n",
    "    initial_values = rand(2)\n",
    "    lower = [0.0, 0.0]\n",
    "    upper = [1.0, 5.0]\n",
    "    inner_optimizer = GradientDescent()\n",
    "    results = optimize(func_qlearning_opt, lower, upper, initial_values, Fminbox(inner_optimizer));\n",
    "end"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### optimization with BlackBoxOptim package"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "using BlackBoxOptim\n",
    "\n",
    "@manipulate for Nₜ in 0:5:200, α in 0:0.05:1, β in 0:0.25:5, Pᵣ in 0:0.05:1\n",
    "    𝐜, 𝐫 = generate_qlearning_data(Nₜ, α, β, Pᵣ)\n",
    "    \n",
    "    func_qlearning_opt(init_values) = func_qlearning(init_values, 𝐜, 𝐫).negll\n",
    "\n",
    "    results = bboptimize(func_qlearning_opt; SearchRange = [(0.0, 1.0), (0.0, 5.0)], NumDimensions = 2);\n",
    "    best_candidate(results)\n",
    "end"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can also compare performances when using different optimizers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "func_qlearning_opt(init_values) = func_qlearning([0.3, 0.4], 𝐜, 𝐫).negll\n",
    "compare_optimizers(func_qlearning_opt; SearchRange = [(0.0, 1.0), (0.0, 5.0)], NumDimensions = 2);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### optimization with JuMP and Ipopt packages"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#The following code block generates error. How can I fix it?\n",
    "\n",
    "using JuMP, Ipopt, ForwardDiff\n",
    "\n",
    "𝐜, 𝐫 = generate_qlearning_data(50, 0.6, 0.7, 0.5)\n",
    "\n",
    "func_qlearning_JuMP(α, β) = func_qlearning((α, β), 𝐜, 𝐫).negll #JuMP needs separate variables, not a list\n",
    "\n",
    "m = Model(Ipopt.Optimizer)\n",
    "register(m, :func_qlearning_JuMP, 2, func_qlearning_JuMP, autodiff=true)\n",
    "\n",
    "@variable(m, 0.0 <= x <= 1.0, start=rand())\n",
    "@variable(m, 0.0 <= y <= 5.0, start=5*rand())\n",
    "@NLobjective(m, Min, func_qlearning_JuMP(x, y))\n",
    "@show optimize!(m)\n",
    "println(\"α = \", value(x), \" β = \", value(y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "@webio": {
   "lastCommId": null,
   "lastKernelId": null
  },
  "hide_input": false,
  "kernelspec": {
   "display_name": "Julia 1.4.1",
   "language": "julia",
   "name": "julia-1.4"
  },
  "language_info": {
   "file_extension": ".jl",
   "mimetype": "application/julia",
   "name": "julia",
   "version": "1.4.1"
  },
  "latex_envs": {
   "LaTeX_envs_menu_present": true,
   "autoclose": false,
   "autocomplete": true,
   "bibliofile": "biblio.bib",
   "cite_by": "apalike",
   "current_citInitial": 1,
   "eqLabelWithNumbers": true,
   "eqNumInitial": 1,
   "hotkeys": {
    "equation": "Ctrl-E",
    "itemize": "Ctrl-I"
   },
   "labels_anchors": false,
   "latex_user_defs": false,
   "report_style_numbering": false,
   "user_envs_cfg": false
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
