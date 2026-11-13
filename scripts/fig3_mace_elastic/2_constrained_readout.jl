# Figure 3, stage 2: the constrained MACE readout.
#
# The 81 readout weights refitted on the training energies and forces (MACE's own loss, force weight 0.1) subject
# to C11, C12, C44 within ±0.1 % of 116.3, 64.8, 30.9 GPa and |P| ≤ 0.001 GPa at a0 = 4.032 Å.
#
# In:   results/fig3/ from stage 1
# Out:  results/fig3/theta_constrained.csv
# Run:  OMP_NUM_THREADS=16 julia --project scripts/fig3_mace_elastic/2_constrained_readout.jl
using PhysicsPriors, LinearAlgebra, DelimitedFiles, Printf

inp = readout_inputs(dirname(input("fig3", "theta0.csv")))
θc  = readout_fit(inp)
writedlm(results("fig3", "theta_constrained.csv"), θc', ',')

names = ("C11", "C12", "C44", "P(a0)")
for (name, θ) in (("MACE as trained", inp.θ0), ("constrained", θc))
    v = inp.base .+ inp.R * θ
    @printf("%-16s %s GPa\n", name, join([@sprintf("%s %9.4f", n, x) for (n, x) in zip(names, v)], "   "))
end
