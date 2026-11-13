# Phonons of FCC Al from supercell force constants.
#
# Two routes to a band structure:
#   native    the model's own Hessian at its current θ and lattice constant (`native_bands`): exact for any θ, one
#             Hessian per θ.
#   undotted  the per-basis Hessian H_k = ∂²B_k/∂r² depends only on geometry, so at a fixed lattice constant it is
#             built once and the dynamical matrix of ANY θ is Σ_k θ_k D_k(q) (`bandpath_basis`, `bands`).  This is
#             what makes phonon stability a linear constraint on θ, and a millisecond check per ensemble member.

# ω² has units eV / (Å² amu) after mass-weighting the Hessian; FREQ_THz ≈ 15.633 converts √ω² to THz.
const _eV_J   = 1.602176634e-19
const _Å_m    = 1.0e-10
const _amu_kg = 1.66053906660e-27
const FREQ_THz = sqrt(_eV_J / (_Å_m^2 * _amu_kg)) / (2π * 1e12)

function _lattice_mat(cell_vectors)
    a, b, c = cell_vectors
    SMatrix{3,3,eltype(a)}(hcat(a, b, c))
end

"""
    bulk_prim_super(element; a=nothing, N_cell=3) -> (sys_prim, sys_super)

The primitive cell and an N_cell³ cubic supercell at lattice constant `a` (Å).  The supercell must be at least
twice the model cutoff across, so that each force constant Φ(i, j+R) is isolated.
"""
function bulk_prim_super(element::Symbol; a=nothing, N_cell::Int=3)
    akw = isnothing(a) ? NamedTuple() : (a = a * u"Å",)
    sys_prim  = bulk(element; akw...)
    sys_super = bulk(element; akw..., cubic=true) * (N_cell, N_cell, N_cell)
    return sys_prim, sys_super
end

"""
    precompute_force_constants(sys_prim, sys_super, model)

The real-space force constants (one Hessian of `sys_super`) and the maps between primitive and supercell atoms
that `dynamical_matrix_from_fc` needs: `p2s_map[i]` is the supercell atom that is the origin-cell image of
primitive atom i, `s2p_map[k]` the primitive class of supercell atom k (both 0-based), `frac_super` the supercell
positions in primitive fractional coordinates.
"""
function precompute_force_constants(sys_prim, sys_super, model)
    H   = ustrip.(hessian(sys_super, model))
    Np  = length(sys_prim)
    Ns  = length(sys_super)
    L_prim   = SMatrix{3,3,Float64}(ustrip.(_lattice_mat(sys_prim.cell.cell_vectors)))
    L_super  = SMatrix{3,3,Float64}(ustrip.(_lattice_mat(sys_super.cell.cell_vectors)))
    Linv_p   = inv(L_prim)
    Linv_s   = inv(L_super)
    frac_prim  = [Linv_p * SVector{3,Float64}(ustrip.(sys_prim[i].position))  for i in 1:Np]
    frac_super = [Linv_p * SVector{3,Float64}(ustrip.(sys_super[k].position)) for k in 1:Ns]
    s2p_map = Vector{Int}(undef, Ns)
    for k in 1:Ns
        matched = false
        for i in 1:Np
            d = mod.(frac_super[k], 1.0) .- mod.(frac_prim[i], 1.0)
            d = d .- round.(d)
            if norm(d) < 1e-6
                s2p_map[k] = i - 1
                matched = true
                break
            end
        end
        matched || error("Supercell atom $k did not match any primitive atom")
    end
    p2s_map = Vector{Int}(undef, Np)
    for i in 1:Np
        candidates = findall(s2p_map .== (i - 1))
        r_ref = SVector{3,Float64}(ustrip.(sys_prim[i].position))
        dists = [norm(SVector{3,Float64}(ustrip.(sys_super[k].position)) .- r_ref)
                 for k in candidates]
        p2s_map[i] = candidates[argmin(dists)] - 1
    end
    masses = [ustrip(sys_prim[i].mass) for i in 1:Np]
    return (; H, frac_prim, frac_super, masses, L=L_prim, L_super, Linv_super=Linv_s, Np, Ns, p2s_map, s2p_map)
end

"""
    dynamical_matrix_from_fc(fc, q_cart) -> Matrix{ComplexF64}

The 3Np × 3Np dynamical matrix at the Cartesian wave vector `q_cart` (Å⁻¹):
D_ij(q) = Σ_k Φ(i, k) exp(i q·R_k) / √(m_i m_j) over the supercell images k of primitive atom j, with R_k the
minimum-image vector in the supercell (the Hessian was computed with periodic boundaries).
"""
function dynamical_matrix_from_fc(fc_data, q_cart::AbstractVector{<:Real})
    (; H, frac_prim, frac_super, masses, L, L_super, Linv_super, Np, Ns, p2s_map, s2p_map) = fc_data
    Dq = zeros(ComplexF64, 3Np, 3Np)
    for i in 0:Np-1, j in 0:Np-1
        dm = zeros(ComplexF64, 3, 3)
        for k in 0:Ns-1
            s2p_map[k + 1] == j || continue
            R_cart = L * (frac_super[k + 1] - frac_prim[j + 1])
            R_sfrac = Linv_super * R_cart
            R_sfrac = R_sfrac .- round.(R_sfrac)
            R_mi    = L_super * R_sfrac
            phase   = dot(q_cart, R_mi)
            eph     = exp(im * phase)
            for α in 1:3, β in 1:3
                dm[α, β] += H[3 * p2s_map[i + 1] + α, 3k + β] * eph
            end
        end
        ms = sqrt(masses[i + 1] * masses[j + 1])
        Dq[3i+1:3i+3, 3j+1:3j+3] .= dm ./ ms
    end
    for i in 1:3Np, j in i:3Np
        Dq[i, j] = (Dq[i, j] + conj(Dq[j, i])) / 2
        Dq[j, i] = conj(Dq[i, j])
    end
    return Dq
end

"""
    fcc_band_path(L; N_per_seg=30) -> (q_list, x_vals, x_ticks, labels, seg_starts)

The FCC path Γ → X → U → L → Γ → K for the primitive lattice matrix `L` (columns are lattice vectors, Å).
`N_per_seg` is the number of q-points per segment: one number, or five (the paper uses [20, 20, 20, 20, 60]: the
last segment, where soft modes tend to sit, is sampled more densely).
"""
function fcc_band_path(L; N_per_seg=30)
    B = 2π * inv(transpose(L))
    frac = (
        Γ = [0.0,   0.0,   0.0],
        X = [0.0,   0.5,   0.5],
        U = [0.25,  0.625, 0.625],
        Lp = [0.5,  0.5,   0.5],
        K = [0.375, 0.75,  0.375],
    )
    pts = (
        Γ = B * frac.Γ,
        X = B * frac.X,
        U = B * frac.U,
        L = B * frac.Lp,
        K = B * frac.K,
    )
    segs   = [(:Γ, :X), (:X, :U), (:U, :L), (:L, :Γ), (:Γ, :K)]
    labels = ["Γ", "X", "U", "L", "Γ", "K"]
    nseg   = N_per_seg isa Integer ? fill(Int(N_per_seg), length(segs)) : collect(Int, N_per_seg)
    length(nseg) == length(segs) ||
        throw(ArgumentError("N_per_seg vector must have $(length(segs)) entries (one per Γ→X→U→L→Γ→K segment)"))
    q_list     = Vector{Float64}[]
    x_vals     = Float64[]
    seg_starts = Bool[]
    x_ticks    = Float64[0.0]
    x = 0.0
    for (s, (l1, l2)) in enumerate(segs)
        q1 = getfield(pts, l1)
        q2 = getfield(pts, l2)
        seg_len = norm(q2 - q1)
        is_last = (s == length(segs))
        n_s     = nseg[s]
        ts = is_last ? range(0.0, 1.0, n_s + 1) :
                       range(0.0, 1.0, n_s + 1)[1:end-1]
        for (ti, t) in enumerate(ts)
            q = q1 .+ t .* (q2 .- q1)
            push!(q_list, q)
            push!(x_vals, x + t * seg_len)
            push!(seg_starts, s == 1 && ti == 1)
        end
        x += seg_len
        push!(x_ticks, x)
    end
    return q_list, x_vals, x_ticks, labels, seg_starts
end

# ── native bands: the model's own Hessian ────────────────────────────────────────────────────────────────────────

"""
    native_bands(model, θ, a; N_cell, N_per_seg, qΓtol=5e-2) -> (; F, x_vals, x_ticks, labels, Np, minω, a)

The phonon bands of θ (THz, 3Np × n_q, imaginary frequencies negative) from the model's own Hessian of an N_cell³
supercell at lattice constant `a`; `minω` is the lowest frequency away from Γ (|q| > qΓtol).  Sets the model's
parameters to θ: pass a private copy of the model when calling from several tasks.
"""
function native_bands(model, θ, a; N_cell, N_per_seg, qΓtol=5e-2)
    set_linear_parameters!(model, θ)
    sp, ss = bulk_prim_super(:Al; a=a, N_cell=N_cell)
    fc = precompute_force_constants(sp, ss, model)
    ql, xv, xt, lb, _ = fcc_band_path(fc.L; N_per_seg=N_per_seg)
    Np = fc.Np; qn = norm.(ql)
    F = Matrix{Float64}(undef, 3Np, length(ql))
    for (iq, q) in enumerate(ql)
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    return (F=F, x_vals=xv, x_ticks=xt, labels=lb, Np=Np,
            minω = minimum(F[:, qn .> qΓtol]), a=a)
end

"The bands of θ at its own relaxed lattice constant."
function native_bands(model, θ; kw...)
    set_linear_parameters!(model, θ)
    return native_bands(model, θ, relax_lattice_constant(model); kw...)
end

"""
    map_with_model_copies(f, model, items; ntasks=Threads.nthreads()) -> Vector

`[f(m, item) for item in items]` on `ntasks` tasks, each with its own `deepcopy(model)` (evaluating a member
mutates the model's parameters).  Results are in input order.
"""
function map_with_model_copies(f, model, items; ntasks=Threads.nthreads())
    n = length(items)
    out = Vector{Any}(undef, n)
    nt  = clamp(ntasks, 1, min(Threads.nthreads(), n))
    @sync for k in 1:nt
        Threads.@spawn begin
            m = deepcopy(model)
            for i in k:nt:n
                out[i] = f(m, items[i])
            end
        end
    end
    return Vector{typeof(out[1])}(out)
end

# ── the undotted per-basis Hessian ───────────────────────────────────────────────────────────────────────────────

function _site_basis_hessian(model, Rs, Zs, z0, ps, st)
    nR = length(Rs)
    x0 = collect(Float64, reinterpret(Float64, Rs))
    tovec(x) = [SVector{3,eltype(x)}(x[3i-2], x[3i-1], x[3i]) for i in 1:nR]
    Bfun(x) = evaluate_basis(model, tovec(x), Zs, z0, ps, st)
    gradflat(x) = vec(ForwardDiff.jacobian(Bfun, x))
    Hflat = ForwardDiff.jacobian(gradflat, x0)
    return reshape(Hflat, length(Bfun(x0)), 3nR, 3nR)
end

"""
    undotted_hessian(sys, V) -> 3N × 3N × n_params

H[:, :, k] = ∂²B_k/∂r²: the Hessian of every basis function of the potential `V` on the cell `sys`.  Parallel over
atoms; the per-thread buffers are summed at the end, so the last bits depend on the number of threads.
"""
function undotted_hessian(sys, V)
    nlist = PairList(sys, cutoff_radius(V)); Nat = length(sys); D = 3; ps = V.ps; st = V.st; m = V.model
    _, Rs0, Zs0, z00 = get_neighbours(sys, V, nlist, 1)
    N_basis = length(evaluate_basis(m, Rs0, Zs0, z00, ps, st))
    Ht = [zeros(D*Nat, D*Nat, N_basis) for _ in 1:Threads.nthreads()]
    Threads.@threads :static for i in 1:Nat
        H = Ht[Threads.threadid()]
        Js, Rs, Zs, z0 = get_neighbours(sys, V, nlist, i)
        # @no_escape: evaluate_basis bump-allocates into this task's Bumper buffer and never rewinds it; unscoped,
        # its temporaries pile up as memory the GC cannot see (>160 GB for one 4×4×4 Al_12 build on 20 threads)
        Hi = @no_escape begin _site_basis_hessian(m, Rs, Zs, z0, ps, st) end
        Ji = (i-1)*D .+ (1:D)
        for (α1, j1) in enumerate(Js), (α2, j2) in enumerate(Js)
            A1 = (α1-1)*D .+ (1:D); A2 = (α2-1)*D .+ (1:D)
            J1 = (j1-1)*D .+ (1:D); J2 = (j2-1)*D .+ (1:D)
            @views for k in 1:N_basis
                blk = Hi[k, A1, A2]
                H[J1,J2,k] .+= blk; H[J1,Ji,k] .-= blk; H[Ji,J2,k] .-= blk; H[Ji,Ji,k] .+= blk
            end
        end
    end
    return sum(Ht)
end

"""
    undotted_hessian_rows(sys_super, V, r; build_threads) -> 3 × 3N × n_params

Only the three rows of atom `r` of the undotted Hessian.  A 1-atom primitive cell reads no others, and only the
neighbours of `r` contribute: for the 684-parameter model on a 4×4×4 cell this is 12.6 MB instead of 3.2 GB.
"""
function undotted_hessian_rows(sys_super, V, r; build_threads)
    nlist = PairList(sys_super, cutoff_radius(V))
    Nat   = length(sys_super); D = 3
    ps, st, m = V.ps, V.st, V.model
    _, Rs0, Zs0, z00 = get_neighbours(sys_super, V, nlist, 1)
    NB = length(evaluate_basis(m, Rs0, Zs0, z00, ps, st))
    Js_r, = get_neighbours(sys_super, V, nlist, r)
    contributors = unique(vcat(r, Js_r))
    nt = clamp(build_threads, 1, min(Threads.nthreads(), length(contributors)))
    @printf("    %d of %d atoms contribute; %d threads, %.2f GB/thread\n",
            length(contributors), Nat, nt, NB*(3length(Js_r))^2*8/1e9); flush(stdout)
    bufs = [zeros(D, D*Nat, NB) for _ in 1:nt]
    @sync for t in 1:nt
        Threads.@spawn begin
            Hl = bufs[t]
            for i in contributors[t:nt:end]
                Js, Rs, Zs, z0 = get_neighbours(sys_super, V, nlist, i)
                Hi = @no_escape begin _site_basis_hessian(m, Rs, Zs, z0, ps, st) end
                Ji = (i-1)*D .+ (1:D); is_ref = (i == r)
                for (α1, j1) in enumerate(Js)
                    hit1 = (j1 == r); (hit1 || is_ref) || continue
                    A1 = (α1-1)*D .+ (1:D)
                    for (α2, j2) in enumerate(Js)
                        A2 = (α2-1)*D .+ (1:D); J2 = (j2-1)*D .+ (1:D)
                        @views for kb in 1:NB
                            blk = Hi[kb, A1, A2]
                            if hit1; Hl[:, J2, kb] .+= blk; Hl[:, Ji, kb] .-= blk; end
                            if is_ref; Hl[:, J2, kb] .-= blk; Hl[:, Ji, kb] .+= blk; end
                        end
                    end
                end
                Hi = nothing
            end
        end
    end
    H = bufs[1]
    for t in 2:nt; H .+= bufs[t]; bufs[t] = zeros(0,0,0); end
    return H
end

"The undotted Hessian of the N_cell³ supercell at `a`, built once and cached in `<model dir>/hessian_cache/`."
function cached_undotted_hessian(m, a, N_cell)
    f = joinpath(m.dir, "hessian_cache", "undotted_Hbasis_$(N_cell)x$(N_cell)x$(N_cell)_a$(round(a;digits=5)).jls")
    if isfile(f)
        c = deserialize(f)
        if abs(c.a_eq - a) < 1e-4; println("  loaded H_basis cache $(basename(f))"); return c.H_basis; end
    end
    mkpath(dirname(f))
    _, sys = bulk_prim_super(:Al; a=a, N_cell=N_cell)
    @printf("  building undotted H_basis at a=%.5f (%d atoms, %d threads) …\n", a, length(sys), Threads.nthreads())
    t = @elapsed Hb = undotted_hessian(sys, m.model)
    @printf("    done in %.1f min\n", t/60)
    serialize(f, (H_basis=Hb, a_eq=a, N_cell=N_cell)); return Hb
end

"""
    bandpath_basis(m, a, N_cell; N_per_seg=20) -> (; Bq, q_list, x_vals, x_ticks, labels, Np, qnorm)

The per-basis dynamical matrices along the band path at lattice constant `a`: `reshape(Bq[iq]*θ, 3Np, 3Np)` is
the dynamical matrix of θ at `q_list[iq]`.  `m` is a loaded model (`load_model`); its parameters are reset to
`m.lin_params`.  Γ is included in the path; the stability checks skip it.
"""
function bandpath_basis(m, a, N_cell; N_per_seg=20)
    model = m.model
    sys_prim, sys_super = bulk_prim_super(:Al; a=a, N_cell=N_cell)
    set_linear_parameters!(model, m.lin_params)
    fc0 = precompute_force_constants(sys_prim, sys_super, model); Np = fc0.Np; N3 = 3*length(sys_super)
    Hb = cached_undotted_hessian(m, a, N_cell)
    n_params = length(m.lin_params)
    q_list, x_vals, x_ticks, labels, _ = fcc_band_path(fc0.L; N_per_seg=N_per_seg)
    Bq = Vector{Matrix{ComplexF64}}(undef, length(q_list))
    for (iq, q) in enumerate(q_list)
        M = Matrix{ComplexF64}(undef, (3Np)^2, n_params)
        for k in 1:n_params
            M[:, k] = vec(dynamical_matrix_from_fc(merge(fc0, (H = reshape(Hb[:, :, k], N3, N3),)), q))
        end
        Bq[iq] = M
    end
    return (; Bq, q_list, x_vals, x_ticks, labels, Np, qnorm = norm.(q_list))
end

"""
    bandpath_basis_rows(model, a; N_cell, N_per_seg, build_threads) -> (; Bq, …, qnorm, a_ref)

`bandpath_basis` for a 1-atom primitive cell from the three Hessian rows it needs (`undotted_hessian_rows`), for
models too large for the full undotted Hessian.
"""
function bandpath_basis_rows(model, a; N_cell, N_per_seg, build_threads)
    sys_prim, sys_super = bulk_prim_super(:Al; a=a, N_cell=N_cell)
    fc = precompute_force_constants(sys_prim, sys_super, model)
    fc.Np == 1 || error("this builder assumes a 1-atom primitive cell; got Np = $(fc.Np)")
    r = fc.p2s_map[1] + 1
    @printf("  building 3-row undotted Hessian at a = %.5f Å (%d atoms) …\n", a, length(sys_super))
    t = @elapsed Hrows = undotted_hessian_rows(sys_super, model, r; build_threads=build_threads)
    @printf("    done in %.1f min\n", t/60); flush(stdout)
    q_list, x_vals, x_ticks, labels, _ = fcc_band_path(fc.L; N_per_seg=N_per_seg)
    NB = size(Hrows, 3)
    Rmi = Vector{SVector{3,Float64}}(undef, fc.Ns)
    for k in 1:fc.Ns
        R_cart  = fc.L * (fc.frac_super[k] - fc.frac_prim[1])
        R_sfrac = fc.Linv_super * R_cart; R_sfrac = R_sfrac .- round.(R_sfrac)
        Rmi[k]  = fc.L_super * R_sfrac
    end
    mass = fc.masses[1]
    Bq = Vector{Matrix{ComplexF64}}(undef, length(q_list))
    Threads.@threads for iq in eachindex(q_list)
        q = q_list[iq]; M = zeros(ComplexF64, 9, NB)
        for k in 1:fc.Ns
            eph = exp(im * dot(q, Rmi[k])); cols = 3(k-1) .+ (1:3)
            @views for kb in 1:NB
                blk = Hrows[:, cols, kb]
                for β in 1:3, α in 1:3; M[3(β-1)+α, kb] += blk[α, β] * eph; end
            end
        end
        M ./= mass
        for kb in 1:NB
            D3 = reshape(@view(M[:, kb]), 3, 3); M[:, kb] = vec((D3 .+ D3') ./ 2)
        end
        Bq[iq] = M
    end
    return (; Bq, q_list, x_vals, x_ticks, labels, Np=fc.Np, qnorm=norm.(q_list), a_ref=a)
end

"The phonon bands of θ on a band-path basis: frequencies in THz, 3Np × n_q, imaginary frequencies negative."
function bands(θ, bp)
    F = Matrix{Float64}(undef, 3bp.Np, length(bp.Bq))
    for iq in eachindex(bp.Bq)
        ev = eigvals(Hermitian(reshape(bp.Bq[iq]*θ, 3bp.Np, 3bp.Np)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    return F
end

"`bands(θ, bp)` in the shape `native_bands` returns, for a θ whose lattice constant is the `a` of `bp`."
function bands_record(θ, bp, a; qΓtol=5e-2)
    F = bands(θ, bp)
    return (F=F, x_vals=bp.x_vals, x_ticks=bp.x_ticks, labels=bp.labels, Np=bp.Np,
            minω = minimum(F[:, bp.qnorm .> qΓtol]), a = a)
end

# ── phonon stability on a band-path basis ────────────────────────────────────────────────────────────────────────

"The squared-frequency threshold, in the units of the dynamical matrix, of a margin given in THz."
omega2_cut(margin_THz) = (margin_THz / FREQ_THz)^2

"The lowest frequency of θ (THz) on the band path, skipping q-points near Γ where the acoustic branches vanish."
function min_freq_stable(θ, bp; qΓtol=5e-2)
    m = Inf
    for iq in eachindex(bp.Bq)
        bp.qnorm[iq] < qΓtol && continue
        ev = eigvals(Hermitian(reshape(bp.Bq[iq]*θ, 3bp.Np, 3bp.Np)))
        m = min(m, minimum(sign.(ev).*sqrt.(abs.(ev)).*FREQ_THz))
    end
    return m
end

"Whether every frequency of θ on the band path is at least `threshold_THz` (returns at the first q below it)."
function is_phonon_stable(θ, bp; threshold_THz, qΓtol=5e-2)
    for iq in eachindex(bp.Bq)
        bp.qnorm[iq] < qΓtol && continue
        ev = eigvals(Hermitian(reshape(bp.Bq[iq]*θ, 3bp.Np, 3bp.Np)))
        if minimum(sign.(ev).*sqrt.(abs.(ev)).*FREQ_THz) < threshold_THz
            return false
        end
    end
    return true
end

"The modes of θ with ω² below `ω2_cut`, as (q index, eigenvector) pairs."
function soft_modes(θ, bp, ω2_cut; qΓtol=5e-2)
    soft = Tuple{Int,Vector{ComplexF64}}[]
    for iq in eachindex(bp.Bq)
        bp.qnorm[iq] < qΓtol && continue
        Fe = eigen(Hermitian(reshape(bp.Bq[iq]*θ, 3bp.Np, 3bp.Np)))
        for ν in 1:3bp.Np
            Fe.values[ν] < ω2_cut && push!(soft, (iq, Fe.vectors[:, ν]))
        end
    end
    return soft
end

"The row c with c·θ = e† D(q) e: the squared frequency of the fixed mode e at q, as a linear function of θ."
cut_row(iq, e, bp) = (c = vec(conj(e)*transpose(e)); real(vec(transpose(c) * bp.Bq[iq])))
