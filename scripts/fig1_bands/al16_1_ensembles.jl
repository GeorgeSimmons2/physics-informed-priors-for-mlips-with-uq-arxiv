# Figure 1, Al_16 (684 parameters), stage 1: the two ensembles of the bottom row.
#
# The POPS cloud is pinned to the lattice constant in closed form (b′·δθ = 0 for every correction), so every draw
# from its hypercube sits at a_eq and one 3-row undotted Hessian screens any of them in milliseconds.
#
# unconstrained  20 draws from the pinned hypercube, no test
# constrained    the same hypercube and seed, keeping only draws with positive curvature b″·θ > 0 and no phonon
#                below −0.05 THz on the band path
#
# In:   models/Al_16_4_6A_3_/
# Out:  results/fig1/al16/ensemble_{unconstrained,constrained}.csv   (20 members each, one per row), bandpath_basis.jls
# Run:  julia --project -t 40 scripts/fig1_bands/al16_1_ensembles.jl
using PhysicsPriors, Random, DelimitedFiles, Serialization, Printf
Random.seed!(1234)

N_cell, N_per_seg = 4, [20, 20, 20, 20, 60]
n_members         = 20
lev_percentile    = 0.5
pin_tol           = 1e-6                         # the largest |b′·δθ| accepted
accept_THz        = -0.05                        # a constrained draw has no phonon below this
build_threads     = 16                           # threads of the Hessian build (0.6 GB each)

m    = load_model(AL16)
prob = regression_problem(m)
a_eq = relax_lattice_constant(m.model)
lat  = lattice_rows(m.model, a_eq)
@printf("%s: %d parameters, %d observations, a_eq = %.6f Å\n", m.name, length(m.lin_params), size(prob.Ap, 1), a_eq); flush(stdout)

cloud = pinned_corrections(prob, lat.b_prime; fraction=lev_percentile, pin_tol)

bp_file = results("fig1", "al16", "bandpath_basis.jls")
bp = isfile(bp_file) ? deserialize(bp_file) :
     (b = bandpath_basis_rows(m.model, a_eq; N_cell, N_per_seg, build_threads); serialize(bp_file, b); b)

eigvecs, bounds = hypercube(cloud)
physical(θ) = positive_curvature(θ, lat) && is_phonon_stable(θ, bp; threshold_THz=accept_THz)

Random.seed!(1234)
unconstrained, _ = sample_hypercube(eigvecs, bounds, m.lin_params; number_of_committee_members=n_members)
Random.seed!(1234)
constrained, _ = rejection_sample_hypercube(eigvecs, bounds, m.lin_params, physical;
                                            number_of_committee_members=n_members, max_attempts=5000 * n_members)

for (tag, mat) in (("unconstrained", unconstrained), ("constrained", constrained))
    w = [min_freq_stable(mat[:, i], bp) for i in 1:n_members]
    @printf("%-14s min ω ∈ [%+.4f, %+.4f] THz, %d of %d below %.2f THz\n", tag, extrema(w)..., count(<(accept_THz), w), n_members, accept_THz)
    writedlm(results("fig1", "al16", "ensemble_$tag.csv"), mat', ',')
end
