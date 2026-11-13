# Figure 2, stage 1: the constrained RLS model.
#
# The RLS fit of the 91-parameter model refitted with its lattice constant pinned, Born stability, and every phonon
# above 0.01 THz at six volumes from a_eq to 1.10 a_eq, so that it stays stable as it expands.
#
# In:   models/Al_12_4_6A_2_/
# Out:  results/fig2/theta_constrained.csv     (and six 5×5×5 undotted Hessians, cached beside the model)
# Run:  julia --project -t 20 scripts/fig2_rls_200K/1_constrained_rls.jl      (1.6 GB of memory per thread)
using PhysicsPriors, LinearAlgebra, Statistics, DelimitedFiles, Printf, OSQP

vol_scales = [1.00, 1.02, 1.04, 1.06, 1.08, 1.10]     # lattice constants, in units of a_eq
N_cell     = 5
margin_THz = 0.01

m    = load_model(AL12)
prob = regression_problem(m)
a_eq = relax_lattice_constant(m.model)
lat  = lattice_rows(m.model, a_eq)
el   = cubic_elastic_rows(m.model, a_eq)
born = born_constraints(el, lat)
bps  = [bandpath_basis(m, s*a_eq, N_cell) for s in vol_scales]
ω2_cut = omega2_cut(margin_THz)

qp, osqp = qp_matrices(prob), OSQP.Model()
fit(cuts) = constrained_fit!(osqp, qp, prob.P, lat.b_prime, vcat(born.rows, cuts),
                             vcat(born.lower, fill(ω2_cut, size(cuts, 1))); settings=OSQP_RLS)
res = fit_with_phonon_cuts(fit, bps, ω2_cut; rounds=1000)
writedlm(results("fig2", "theta_constrained.csv"), res.θ, ',')

rmse(θ) = sqrt(mean((prob.Ap*(prob.P*θ) .- prob.Yw).^2))
for (name, θ) in (("RLS", m.lin_params), ("constrained RLS", res.θ))
    ω = join([@sprintf("@%d%% %+.3f", round(Int, 100s), min_freq_stable(θ, bp)) for (s, bp) in zip(vol_scales, bps)], "  ")
    @printf("%-16s min ω (THz) %s   weighted RMSE %.4g\n", name, ω, rmse(θ))
end
@printf("%d phonon cuts, ‖Δθ‖ = %.3g\n", res.n_cuts, norm(res.θ - m.lin_params))
