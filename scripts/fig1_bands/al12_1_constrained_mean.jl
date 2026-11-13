# Figure 1, Al_12 (91 parameters), stage 1: the constrained mean model.
#
# The RLS fit refitted with its lattice constant pinned (b′·θ = 0), Born stability, and every phonon on the band
# path at least 0.15 THz: cutting planes are added for the soft modes until none is left.
#
# In:   models/Al_12_4_6A_2_/
# Out:  results/fig1/al12/theta_mean.csv        (and the 4×4×4 undotted Hessian, cached beside the model)
# Run:  julia --project -t 20 scripts/fig1_bands/al12_1_constrained_mean.jl
using PhysicsPriors, LinearAlgebra, DelimitedFiles, Printf, OSQP

N_cell, N_per_seg = 4, [20, 20, 20, 20, 60]      # supercell of the force constants; q-points per path segment
cut_margin_THz    = 0.15                         # every phonon is pushed above this
max_cuts          = 40                           # rounds of cutting planes

m    = load_model(AL12)
prob = regression_problem(m)
a_eq = relax_lattice_constant(m.model)
el   = cubic_elastic_rows(m.model, a_eq)
lat  = lattice_rows(m.model, a_eq)
born = born_constraints(el, lat)
bp   = bandpath_basis(m, a_eq, N_cell; N_per_seg)
ω2_cut = omega2_cut(cut_margin_THz)
@printf("%s: %d parameters, a_eq = %.5f Å, RLS min ω = %.3f THz\n", m.name, length(m.lin_params), a_eq, min_freq_stable(m.lin_params, bp))

qp, osqp = qp_matrices(prob), OSQP.Model()
fit(cuts) = constrained_fit!(osqp, qp, prob.P, lat.b_prime, vcat(born.rows, cuts),
                             vcat(born.lower, fill(ω2_cut, size(cuts, 1))); settings=OSQP_AL12)
θ_mean = fit_with_phonon_cuts(fit, [bp], ω2_cut; rounds=max_cuts + 1).θ

writedlm(results("fig1", "al12", "theta_mean.csv"), θ_mean, ',')
@printf("constrained mean: C11 = %.1f, C12 = %.1f, C44 = %.1f GPa, b′·θ = %.1e, min ω = %.3f THz\n",
        elastic_constants(el, θ_mean)..., dot(lat.b_prime, θ_mean), min_freq_stable(θ_mean, bp))
