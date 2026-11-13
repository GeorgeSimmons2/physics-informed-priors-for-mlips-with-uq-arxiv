# Figure 3, stage 3: the POPS clouds of the MACE readout.
#
# constrained    one constrained fit per high-leverage observation (the fit of stage 2 that also reproduces that
#                energy or force component), about the constrained readout θc
# unconstrained  the closed-form POPS corrections about MACE's trained readout θ0, for the same observations
#
# In:   results/fig3/ from stages 1 and 2
# Out:  results/fig3/clouds.jls        Θ_con, Θ_unc (81 × N, one member per column), θc, θ0, R, lo, hi
# Run:  julia --project -t 16 scripts/fig3_mace_elastic/3_pops_clouds.jl
using PhysicsPriors, LinearAlgebra, DelimitedFiles, Serialization
BLAS.set_num_threads(1)                          # the Julia threads own the parallelism

dir = dirname(input("fig3", "XF_train.csv"))
inp = readout_inputs(dir)
θc  = vec(readdlm(input("fig3", "theta_constrained.csv"), ','))
XF  = readdlm(joinpath(dir, "XF_train.csv"), ',', Float64)
yF  = vec(readdlm(joinpath(dir, "yF_train.csv"), ',', Float64))

clouds = readout_clouds(inp, θc, XF, yF)
print(clouds.summary)
serialize(results("fig3", "clouds.jls"), (; clouds.Θ_con, clouds.Θ_unc, θc, inp.θ0, inp.R, inp.lo, inp.hi))
