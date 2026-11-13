# Figure 2, stage 2: 200 K NPT molecular dynamics with one parameter vector, and the RDF of the run.
#
# 4×4×4 cubic cell (256 atoms), zero pressure, Langevin thermostat with a Monte Carlo barostat, 1 fs time step,
# 10,000 steps of equilibration and 20,000 of production.  Both models start from the same seed.
#
# In:   models/Al_12_4_6A_2_/, and for the constrained model results/fig2/theta_constrained.csv
# Out:  results/fig2/<model>/md_trajectory.extxyz, a_200K.csv, rdf_200K.csv
# Run:  julia --project -t 4 scripts/fig2_rls_200K/2_npt.jl rls
#       julia --project -t 4 scripts/fig2_rls_200K/2_npt.jl constrained           (about 10 h each on 4 cores)
using PhysicsPriors, DelimitedFiles, Printf

tag = only(ARGS)                                 # rls | constrained
T_K, seed = 200, 1234

m = load_model(AL12; design=false)
θ = tag == "rls" ? m.lin_params : vec(readdlm(input("fig2", "theta_constrained.csv"), ','))

PhysicsPriors.set_linear_parameters!(m.model, θ)
a0 = relax_lattice_constant(m.model)
@printf("%s: relaxed a0 = %.6f Å at 0 K; running %d K NPT\n", tag, a0, T_K); flush(stdout)

sys = run_npt(m.model, θ; a0, T_K, seed)

trajectory = results("fig2", tag, "md_trajectory.extxyz")
write_trajectory(trajectory, sys)
a, a_std = lattice_constant_from_volume(sys)
open(results("fig2", tag, "a_200K.csv"), "w") do io
    println(io, "a_Ang,a_std_Ang"); @printf(io, "%.6f,%.6f\n", a, a_std)
end
r, g = rdf_from_trajectory(trajectory)
writedlm(results("fig2", tag, "rdf_200K.csv"), vcat(["r_Ang" "g_r"], hcat(r, g)), ',')
@printf("a(%d K) = %.5f ± %.5f Å (%+.3f%% from a0)\n", T_K, a, a_std, 100*(a - a0)/a0)
