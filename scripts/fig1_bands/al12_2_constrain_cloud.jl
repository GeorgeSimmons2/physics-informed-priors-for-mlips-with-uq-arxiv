# Figure 1, Al_12, stage 2: the constrained POPS cloud.
#
# For each of the 73,479 observations in the top half by leverage: the constrained fit of stage 1 that also
# reproduces that observation exactly.  One quadratic programme per observation, with its own cutting planes.
#
# In:   models/Al_12_4_6A_2_/ (and the Hessian cache of stage 1)
# Out:  results/fig1/al12/cloud_constrained.jls    Θ (91 × N): the members that reached the 0.15 THz margin
# Run:  julia --project -t 42 scripts/fig1_bands/al12_2_constrain_cloud.jl        (about 4 h on 42 cores)
using PhysicsPriors, LinearAlgebra, Serialization, Printf, OSQP

N_cell, N_per_seg = 4, [20, 20, 20, 20, 60]
cut_margin_THz    = 0.15
max_cuts          = 40
lev_percentile    = 0.5                          # the fraction of observations, by leverage, that form the cloud
BLAS.set_num_threads(1)                          # the Julia threads own the parallelism

m    = load_model(AL12)
prob = regression_problem(m)
a_eq = relax_lattice_constant(m.model)
el   = cubic_elastic_rows(m.model, a_eq)
lat  = lattice_rows(m.model, a_eq)
born = born_constraints(el, lat)
bp   = bandpath_basis(m, a_eq, N_cell; N_per_seg)
ω2_cut = omega2_cut(cut_margin_THz)

selected = top_leverage_rows(prob, lev_percentile)
N = length(selected)
@printf("%d of %d observations selected; %d threads\n", N, size(prob.Ap, 1), Threads.nthreads()); flush(stdout)

qp    = qp_matrices(prob)
osqps = [OSQP.Model() for _ in 1:Threads.nthreads()]
Θ     = zeros(length(m.lin_params), N)
minω  = fill(NaN, N)
done  = Threads.Atomic{Int}(0)
Threads.@threads for k in 1:N
    i, osqp = selected[k], osqps[Threads.threadid()]
    fit(cuts) = constrained_fit!(osqp, qp, prob.P, lat.b_prime, vcat(born.rows, cuts),
                                 vcat(born.lower, fill(ω2_cut, size(cuts, 1)));
                                 interpolate=(prob.Ap[i, :], prob.Yw[i]), settings=OSQP_AL12)
    θ = fit_with_phonon_cuts(fit, [bp], ω2_cut; rounds=max_cuts + 1, refit_on_last=false).θ
    Θ[:, k] = θ; minω[k] = min_freq_stable(θ, bp)
    d = Threads.atomic_add!(done, 1) + 1
    d % 5000 == 0 && (@printf("  %d / %d\n", d, N); flush(stdout))
end

at_margin = (!isnan).(minω) .& (minω .>= cut_margin_THz - 1e-6)
serialize(results("fig1", "al12", "cloud_constrained.jls"), (; Θ=Θ[:, at_margin], selected=selected[at_margin], minω=minω[at_margin]))
@printf("%d of %d members reached the %.2f THz margin\n", count(at_margin), N, cut_margin_THz)
