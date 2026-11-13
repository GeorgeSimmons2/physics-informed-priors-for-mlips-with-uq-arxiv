# Figure 1, Al_12, stage 3: the constrained ensemble.
#
# The hypercube of the constrained cloud about the constrained mean, rejection-sampled: a draw is kept if it is
# Born stable, its lattice constant has not drifted, and its phonons are above the margin on the whole band path.
#
# In:   results/fig1/al12/theta_mean.csv, cloud_constrained.jls
# Out:  results/fig1/al12/ensemble_constrained.csv     30 members, one per row
# Run:  julia --project -t 8 scripts/fig1_bands/al12_3_constrained_ensemble.jl
using PhysicsPriors, Random, DelimitedFiles, Serialization, Printf
Random.seed!(1234)

N_cell, N_per_seg = 4, [20, 20, 20, 20, 60]
cut_margin_THz    = 0.15
n_members         = 30

m    = load_model(AL12; design=false)
a_eq = relax_lattice_constant(m.model)
el   = cubic_elastic_rows(m.model, a_eq)
lat  = lattice_rows(m.model, a_eq)
born = born_constraints(el, lat)
bp   = bandpath_basis(m, a_eq, N_cell; N_per_seg)

θ_mean = vec(readdlm(input("fig1", "al12", "theta_mean.csv"), ','))
Θ      = deserialize(input("fig1", "al12", "cloud_constrained.jls")).Θ
eigvecs, bounds = hypercube(Matrix(Θ' .- θ_mean'))
@printf("cloud of %d members → hypercube with %d directions\n", size(Θ, 2), size(eigvecs, 2)); flush(stdout)

physical(θ) = is_born_stable(born, θ) && lattice_drift_ok(θ, lat, θ_mean) &&
              min_freq_stable(θ, bp) >= cut_margin_THz - 1e-6
ensemble, _ = rejection_sample_hypercube(eigvecs, bounds, θ_mean, physical;
                                         number_of_committee_members=n_members, max_attempts=5_000_000)
writedlm(results("fig1", "al12", "ensemble_constrained.csv"), ensemble', ',')
