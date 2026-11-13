# The MACE readout (the Julia half of Figures 3 and 4).
#
# MACE's energy is linear in its readout weights θ (81 of them here: E = E_frozen + D·θ), so the same machinery
# applies: python/mace_readout.py writes the basis D of every training energy and force component and the rows
# that give C11, C12, C44 and the pressure at the target lattice constant; here θ is refitted on the training set
# subject to those constraints, and POPS committees are built for the constrained and the unconstrained readout.
#
# The loss has MACE's own form, per-atom energy MSE + w_F × force-component MSE with w_F = 0.1.  The last column
# of D (the per-atom shift) is held at its trained value.  Problems are solved in z = S(θ − θ_ref) with S the
# column RMS of D/N, in the eigenbasis of the scaled Hessian, keeping eigenvalues above RTOL of the largest.

const GPA = 160.21766208                              # eV/Å³ → GPa
const C_TARGET = [116.3, 64.8, 30.9]                  # C11, C12, C44 of Al (GPa) at a0 = 4.032 Å

"""
    readout_inputs(dir; TOL=1e-3, P_TOL=1e-3)

What `python/mace_readout.py` wrote to `dir`: the trained weights θ0, the constraint rows `R` (C11, C12, C44 and
the pressure P at a0, in GPa) with their windows `lo ≤ R·θ ≤ hi` (C_ij within ±TOL of the targets, |P| ≤ P_TOL
GPa), the per-atom energy rows (XE, yE) and the force rows as normal equations (GtG, Gty, nF).
"""
function readout_inputs(dir; TOL=1e-3, P_TOL=1e-3)
    rd(f) = readdlm(joinpath(dir, f), ',', Float64)
    row(f) = vec(rd(f))
    meta = Dict(String(s[1]) => parse(Float64, s[2]) for s in split.(readlines(joinpath(dir, "meta.txt"))))
    a0, V = meta["a0"], meta["V"]
    θ0, fz = row("theta0.csv"), row("frozen.csv")     # fz: the frozen part's dE/da, d²E/da², d²E/dε11², /dε11dε22, /dε44²
    R = [GPA / V .* row("d2D_de11.csv")'
         GPA / V .* row("d2D_de12.csv")'
         GPA / V .* row("d2D_de44.csv")'
         GPA / (3a0^2) .* row("dD_da.csv")']
    base = [GPA / V * fz[3], GPA / V * fz[4], GPA / V * fz[5], GPA / (3a0^2) * fz[1]]
    lo = [C_TARGET .* (1 - TOL); -P_TOL] .- base
    hi = [C_TARGET .* (1 + TOL);  P_TOL] .- base
    fst = rd("fstats_train.csv")
    return (; θ0, a0, V, R, base, lo, hi, XE = rd("XE_train.csv"), yE = row("yE_train.csv"),
            GtG = rd("GtG_train.csv"), Gty = row("Gty_train.csv"), yty = fst[1], nF = fst[2])
end

"""
    readout_fit(inp; fw=0.1, constrained=true, RTOL=1e-12) -> θ

Refit the readout weights on the training energies and forces (force weight `fw`); with `constrained`, subject to
the elastic-constant and pressure windows of `readout_inputs`.
"""
function readout_fit(inp; fw=0.1, constrained=true, RTOL=1e-12)
    (; θ0, R, lo, hi, GtG, Gty, nF) = inp
    X, y = inp.XE, inp.yE
    n = length(y); p = length(θ0)
    k = 1:p-1                                         # the shift column is held at θ0
    S = vec(sqrt.(mean(X[:, k] .^ 2, dims=1)))
    H = X' * X / n + fw * GtG / nF
    g = X' * y / n + fw * Gty / nF
    Hs = Symmetric(H[k, k] ./ (S * S'))
    qs = -(g[k] .- H[k, :] * θ0) ./ S
    λ, U = eigen(Hs)
    keep = λ .> RTOL * maximum(λ)
    Ur, λr = U[:, keep], λ[keep]
    qr = Ur' * qs
    if constrained
        m = OSQP.Model()
        OSQP.setup!(m; P=sparse(Diagonal(λr)), q=qr, A=sparse((R[:, k] ./ S') * Ur), l=lo .- R * θ0, u=hi .- R * θ0,
                    eps_abs=1e-9, eps_rel=1e-9, max_iter=200_000, polish=true, verbose=false)
        res = OSQP.solve!(m)
        res.info.status == :Solved || error("OSQP status $(res.info.status)")
        z = Ur * res.x
    else
        z = Ur * (-qr ./ λr)
    end
    θ = copy(θ0); θ[k] .+= z ./ S
    return θ
end

"""
    readout_clouds(inp, θc, XF, yF; fw=0.1, lev_percentile=0.5, RTOL=1e-12) -> (; Θ_con, Θ_unc, summary)

The POPS clouds of the readout, one member per column.  Observations are every training energy (per atom) and
force component; the top `lev_percentile` by leverage are kept.

  Θ_con   for each kept observation, the constrained fit that also reproduces that observation (one QP each,
          solved about the constrained mean θc)
  Θ_unc   the closed-form POPS corrections about MACE's trained weights θ0, for the same observations
"""
function readout_clouds(inp, θc, XF, yF; fw=0.1, lev_percentile=0.5, RTOL=1e-12)
    (; θ0, R, lo, hi, XE, yE) = inp
    p = length(θ0); k = 1:p-1
    nE, nF = length(yE), length(yF)
    A = vcat(XE, XF); y = vcat(yE, yF); w = vcat(fill(1 / nE, nE), fill(fw / nF, nF))
    isE = vcat(trues(nE), falses(nF))
    H = A' * (w .* A); g = A' * (w .* y)
    S = vec(sqrt.(mean(XE[:, k] .^ 2, dims=1)))
    Hs = Symmetric(H[k, k] ./ (S * S'))
    λ, U = eigen(Hs)
    keepdir = λ .> RTOL * maximum(λ)
    Ur, λr = U[:, keepdir], λ[keepdir]
    Hinv(v) = Ur * ((Ur' * v) ./ λr)                    # H_s⁻¹ on the kept subspace

    Ak = sparse(R[:, k] ./ S')
    qc = -(g[k] .- H[k, :] * θc) ./ S
    Psp = sparse(Diagonal(λr))
    qr = Ur' * qc
    AkU = Matrix(Ak) * Ur
    function osqp_solve(Aqp, l, u)                       # Aqp in ζ (z = Ur ζ); returns z
        m = OSQP.Model()
        OSQP.setup!(m; P=Psp, q=qr, A=sparse(Aqp), l=l, u=u, eps_abs=1e-9, eps_rel=1e-9,
                    max_iter=200_000, polish=true, verbose=false, adaptive_rho_interval=25)
        res = OSQP.solve!(m)
        Ur * res.x, res.info.status
    end
    z_mean, _ = osqp_solve(AkU, lo .- R * θc, hi .- R * θc)   # the mean model's QP rebuilt from the rows: z = 0

    # leverage → kept observations
    B = A[:, k] ./ S'                                   # rows in z-space
    lev = w .* vec(sum(B .* Hinv(Matrix(B'))', dims=2))
    thresh = quantile(lev, 1 - lev_percentile)
    kept = findall(>=(thresh), lev)
    N = length(kept)

    # constrained cloud: one QP per kept observation
    Θ_con = zeros(p, N); status = fill(:unset, N)
    Threads.@threads for j in 1:N
        r = kept[j]
        Aqp = vcat(AkU, reshape(Ur' * B[r, :], 1, :))
        e = y[r] - dot(A[r, :], θc)
        z, st = osqp_solve(Aqp, vcat(lo .- R * θc, e), vcat(hi .- R * θc, e))
        θ = copy(θc); θ[k] .+= z ./ S
        Θ_con[:, j] = θ; status[j] = st
    end
    ok = status .== :Solved
    Θ_con = Θ_con[:, ok]

    # unconstrained cloud: closed-form corrections about θ0
    Bk = B[kept, :]'                                     # (p−1) × N
    HiB = Hinv(Bk)
    e0 = y[kept] .- A[kept, :] * θ0
    Δz = HiB .* (e0 ./ vec(sum(Bk .* HiB, dims=1)))'
    Θ_unc = repeat(θ0, 1, N); Θ_unc[k, :] .+= Δz ./ S

    summary = @sprintf("""
        POPS clouds of the MACE readout
        observations: %d energies + %d force components; top %.0f%% by leverage → %d kept (%d energies)
        constrained:   %d of %d QPs solved
        unconstrained: %d closed-form members about θ0
        mean-model check: the QP rebuilt from the rows moves θc by ‖z‖ = %.2e
        """, nE, nF, 100lev_percentile, N, count(isE[kept]), count(ok), N, N, norm(z_mean))
    return (; Θ_con, Θ_unc, summary)
end
