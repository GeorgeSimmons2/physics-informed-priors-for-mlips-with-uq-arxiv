# The constrained fit: the model's own least-squares problem, minimised subject to physical constraints that are
# linear in θ.  Solved with OSQP in the preconditioned variable x = Pθ.
#
#   minimise    ½ xᵀ(ÃᵀÃ + λPᵀP)x − (Ãᵀỹ)ᵀx
#   subject to  b′·θ = 0                      the lattice constant stays at a_eq
#               rows·θ ≥ lower                Born stability, and one row per phonon cut
#               a_i·θ = y_i                   (optional) reproduce observation i exactly: a POPS member
#
# Phonon stability is not linear in θ, but "mode e at wave vector q has ω² ≥ cut" is (`cut_row`).  So the fit is
# repeated, each time adding a cut for every mode found soft, until none is left (`fit_with_phonon_cuts`).

"The quadratic form of the fit in x = Pθ."
qp_matrices(prob) = (; H = sparse(prob.Ap'*prob.Ap .+ prob.λ.*(prob.P'*prob.P)), q = -(prob.Ap'*prob.Yw))

"OSQP settings of the Al_12 committee fits (Figure 1)."
const OSQP_AL12 = (; max_iter=4_000_000, check_termination=25, eps_abs=1e-6, eps_rel=1e-6, adaptive_rho_interval=25)
"OSQP settings of the constrained RLS fit (Figure 2)."
const OSQP_RLS  = (; max_iter=1_000_000, eps_abs=1e-8, eps_rel=1e-8, adaptive_rho_interval=25)

"""
    constrained_fit!(osqp, qp, P, b_prime, rows, lower; interpolate=nothing, settings) -> (θ, status)

One constrained fit.  `rows` and `lower` are in θ-space; `interpolate = (a_i, y_i)` adds the equality a_i·x = y_i
(a row of Ã and its target).  `adaptive_rho_interval` is part of every settings preset: it makes OSQP's step-size
schedule independent of wall-clock time, so a rerun follows the same iterates.
"""
function constrained_fit!(osqp, qp, P, b_prime, rows, lower; interpolate=nothing, settings)
    A = vcat(sparse(b_prime'/P), sparse(rows/P))
    l = vcat([0.0], lower); u = vcat([0.0], fill(Inf, length(lower)))
    if interpolate !== nothing
        a_i, y_i = interpolate
        A = vcat(sparse(a_i'), A); l = vcat([y_i], l); u = vcat([y_i], u)
    end
    OSQP.setup!(osqp; P=qp.H, q=qp.q, A=A, l=l, u=u, verbose=false, settings...)
    r = OSQP.solve!(osqp)
    return P \ r.x, r.info.status
end

"""
    fit_with_phonon_cuts(fit, bps, ω2_cut; rounds, refit_on_last=true, qΓtol=5e-2)
        -> (; θ, status, n_cuts, n_rounds)

Cutting planes.  `fit(cut_rows)` returns `(θ, status)` of the constrained fit with those extra rows held at
`≥ ω2_cut`; after each fit every soft mode of θ on every band-path basis in `bps` (one per volume) becomes a new
row.  Stops when no mode is soft or after `rounds` checks; with `refit_on_last=false` the last check adds no
cuts.  `n_rounds` counts the checks that found a soft mode.
"""
function fit_with_phonon_cuts(fit, bps, ω2_cut; rounds, refit_on_last=true, qΓtol=5e-2)
    cut_rows = zeros(0, size(first(bps).Bq[1], 2))
    θ, status = fit(cut_rows)
    n_cuts = 0; n_rounds = 0
    for it in 1:rounds
        soft = [(v, iq, e) for v in eachindex(bps) for (iq, e) in soft_modes(θ, bps[v], ω2_cut; qΓtol=qΓtol)]
        isempty(soft) && break
        n_rounds += 1
        (!refit_on_last && it == rounds) && break
        for (v, iq, e) in soft; cut_rows = vcat(cut_rows, cut_row(iq, e, bps[v])'); end
        n_cuts += length(soft)
        θ, status = fit(cut_rows)
    end
    return (; θ, status, n_cuts, n_rounds)
end
