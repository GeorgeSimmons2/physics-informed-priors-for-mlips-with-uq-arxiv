# check_rejection_rate.jl — acceptance rate of the constrained hypercube rejection sampler behind the
# published four-panel figure, re-measured for Al_12 and Al_16 from each study's own point cloud and predicate.
#
# For each study the script rebuilds exactly what the sampling script built — the POPS delta cloud, its
# hypercube (POPSRegression.hypercube, unclipped), and the acceptance predicate — and then measures two things:
#   (a) the published draw: seed 1234, rejection_sample_hypercube for the published member count (30 for Al_12,
#       20 for Al_16), which must reproduce the recorded acceptance (al12: published/al12/hypercube/
#       hypercube_summary.csv, 7.177 % full cloud / 32.26 % forest; al16: the % printed by al16/01) if the
#       cloud, box and predicate are the ones that were used;
#   (b) a fixed number of proposals (N_PROP, default 20000) with the funnel broken down by stage, giving the
#       acceptance with a binomial standard error that does not depend on the stopping rule of (a).
# It also checks that the published Al_12 members (committee_rejection_full_cloud.csv, the first 30 rows of
# which are the four-panel figure's constrained ensemble) all satisfy the predicate.
#
#   Al_12  cloud: $OUT/al12/cloud/committee_stable.jls (al12/02; the published tree does not carry the 73,390-
#          member cloud, so this needs a `saved` route) and the 30-member forest (published/al12/committee).
#          Predicate (al12/03): Born rows ≥ [0.1, 1, 0.1, 1e-9], |Δa_eq| ≤ 10 %, min ω ≥ 0.15 THz on the band path.
#   Al_16  cloud: the pinned closed-form corrections of al16/01 (rebuilt here from the design matrix, top-50 %
#          leverage).  Predicate: b″·θ > 0 and min ω ≥ ACCEPT_TOL (−0.05 THz, the published run) on the cached
#          band path (BPCACHE; al16/01 writes it, the published model tree may carry it).
#
# Run:  julia --project -t 8 four_panel/check_rejection_rate.jl
# Env:  OUT (repro/saved), N_PROP (20000), WHICH (al12,al16), BPCACHE, ACE_MODELS (via model_dir)
# Out:  $OUT/four_panel/rejection_rate.txt

include(joinpath(@__DIR__, "..", "lib", "bandpath.jl"))   # brings in ACEWorkflow, which exports the POPS sampler
using Random, LinearAlgebra, Statistics, Serialization, DelimitedFiles, Printf, ForwardDiff, Unitful

N_PROP = parse(Int, get(ENV, "N_PROP", "20000"))
WHICH  = split(get(ENV, "WHICH", "al12,al16"), ",")
ROUT   = get(ENV, "OUT", normpath(joinpath(@__DIR__, "..", "repro", "saved")))
PUB    = normpath(joinpath(@__DIR__, "..", "published"))
OUTDIR = joinpath(ROUT, "four_panel"); mkpath(OUTDIR)
LINES  = String[]
say(s) = (println(s); push!(LINES, s); flush(stdout))

firstfile(cands...) = (i = findfirst(isfile, cands); i === nothing ? nothing : cands[i])
se(k, n) = n == 0 ? NaN : sqrt(k / n * (1 - k / n) / n)

"""Draw N proposals from the box, push each through `stages` in order; return the survivor count per stage."""
function funnel(heig, hb, centre, stages, N)
    lo, hi = hb[1, :], hb[2, :]
    surv = zeros(Int, length(stages))
    for _ in 1:N
        θ = centre .+ heig * (lo .+ (hi .- lo) .* rand(length(lo)))
        for (i, (_, f)) in enumerate(stages)
            f(θ) || break
            surv[i] += 1
        end
    end
    surv
end

function report(label, stages, surv, N, recorded)
    say(@sprintf("  %-28s %8s %8s", "stage", "pass", "of prev"))
    prev = N
    for (i, (nm, _)) in enumerate(stages)
        say(@sprintf("  %-28s %8d %7.2f%%", nm, surv[i], 100surv[i] / max(prev, 1)))
        prev = surv[i]
    end
    k = surv[end]
    say(@sprintf("  acceptance from %d proposals: %.3f%% ± %.3f%%   (recorded in the published run: %s)",
                 N, 100k / N, 100se(k, N), recorded))
end

"""Seeded reproduction of the published draw: the acceptance the sampler itself reports."""
function published_draw(heig, hb, centre, stages, n_members; seed=1234, max_attempts=5_000_000)
    n_prop = Ref(0)
    pred = θ -> begin
        n_prop[] += 1
        for (_, f) in stages
            f(θ) || return false
        end
        true
    end
    Random.seed!(seed)
    rejection_sample_hypercube(heig, hb, centre, pred; number_of_committee_members=n_members, max_attempts=max_attempts)
    n_members, n_prop[]
end

# ═══════════════════════════════════════════════════════════════════════════
if "al12" in WHICH
    say("══ Al_12_4_6A_2_  (al12/03_hypercube_rejection.jl predicate) ══")
    result = load_model(:Al, 12, 4, 6, 2; dataset_name="")
    model = result.model; n_params = length(result.lin_params)
    a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)
    H_el = elastic_hessian_basis(model; element=:Al, a=a_eq)
    c11_0 = reshape(H_el, 36, n_params)[1, :]; c12_0 = reshape(H_el, 36, n_params)[7, :]; c44_0 = reshape(H_el, 36, n_params)[22, :]
    lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(ACEWorkflow.Elasticity.reference_system(:Al; a=a), model))
    b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
    b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
    born_rows  = vcat(c44_0', (c11_0 .- c12_0)', (c11_0 .+ 2 .* c12_0)', b_double_prime')
    born_lower = [0.1, 1.0, 0.1, 1e-9]
    bp = bandpath_Dk(result, model, :Al, a_eq, 4; N_per_seg=[20, 20, 20, 20, 60])
    cut, qΓtol = 0.15, 5e-2
    say(@sprintf("a_eq = %.5f Å; band path %d q-points; %d parameters", a_eq, length(bp.Bq), n_params))

    cdir = dirname(firstfile(joinpath(ROUT, "al12", "committee", "theta_mean.csv"), joinpath(PUB, "al12", "committee", "theta_mean.csv")))
    θ_mean = vec(readdlm(joinpath(cdir, "theta_mean.csv"), ','))
    K_ref = dot(θ_mean, b_double_prime)
    stages = [("Born rows ≥ lower",     θ -> all(born_lower .<= born_rows * θ)),
              ("|Δa_eq| ≤ 10 %",        θ -> abs(dot(b_prime, θ .- θ_mean) / K_ref) <= 0.1),
              ("min ω ≥ 0.15 THz",      θ -> min_freq_stable(θ, bp; qΓtol=qΓtol) >= cut - 1e-6)]

    recorded = Dict{String,String}()
    sumf = joinpath(PUB, "al12", "hypercube", "hypercube_summary.csv")
    if isfile(sumf)
        for r in eachrow(readdlm(sumf, ','; skipstart=1))
            recorded[String(r[1])] = @sprintf("%.3f%%", r[6])
        end
    end

    clouds = Tuple{String,Matrix{Float64},Int,String}[]
    full = joinpath(ROUT, "al12", "cloud", "committee_stable.jls")
    if isfile(full)
        push!(clouds, ("full cloud", Matrix{Float64}(deserialize(full).Θ), 30, get(recorded, "full_cloud", "n/a")))
    else
        say("  full cloud $(full) not found (al12/02 output; not in published/) — the four-panel ensemble was drawn from it; " *
            "only the 30-member forest is checked")
    end
    push!(clouds, ("forest 30", Matrix{Float64}(Matrix(readdlm(joinpath(cdir, "committee_repaired.csv"), ',')')), 30,
                   get(recorded, "forest_30", "n/a")))

    for (label, Θ, n_members, rec) in clouds
        deltas = Matrix(Θ' .- θ_mean')
        heig, hb = hypercube(deltas)
        w = hb[2, :] .- hb[1, :]
        say(@sprintf("\n[%s] %d cloud members → hypercube with %d directions (width mean %.4g, max %.4g)",
                     label, size(Θ, 2), size(heig, 2), mean(w), maximum(w)))
        acc, n_prop = published_draw(heig, hb, θ_mean, stages, n_members)
        say(@sprintf("  published draw (seed 1234, %d members): %d proposals → %.3f%% acceptance   (recorded: %s)%s",
                     n_members, n_prop, 100acc / n_prop, rec,
                     label == "forest 30" ? "  [al12/03 drew this one second, on the continued stream: compare the fixed-N estimate]" : ""))
        Random.seed!(4321)
        t = @elapsed surv = funnel(heig, hb, θ_mean, stages, N_PROP)
        report(label, stages, surv, N_PROP, rec)
        say(@sprintf("  [%.1f s]", t))
    end

    memf = joinpath(PUB, "al12", "hypercube", "committee_rejection_full_cloud.csv")
    if isfile(memf)
        M = readdlm(memf, ',')
        ok = [all(f(vec(M[i, :])) for (_, f) in stages) for i in 1:size(M, 1)]
        mf = [min_freq_stable(vec(M[i, :]), bp; qΓtol=qΓtol) for i in 1:size(M, 1)]
        say(@sprintf("\n  published members (%s): %d / %d satisfy the predicate; min ω ∈ [%.4f, %.4f] THz (first 30 rows are the four-panel constrained ensemble)",
                     basename(memf), count(ok), length(ok), minimum(mf), maximum(mf)))
    end
end

# ═══════════════════════════════════════════════════════════════════════════
if "al16" in WHICH
    say("\n══ Al_16_4_6A_3_  (al16/01_pinned_ensembles.jl predicate) ══")
    ACCEPT_TOL = parse(Float64, get(ENV, "ACCEPT_TOL", "-0.05"))
    N_MEMBERS  = parse(Int, get(ENV, "N_MEMBERS", "20"))
    LEV_PCT    = 0.5
    result = load_model(:Al, 16, 4, 6, 3; dataset_name="")
    model, lin_params = result.model, result.lin_params
    P, W, Y = result.P, result.W, result.Y
    Ap = Diagonal(W) * result.A / P; Yw = W .* Y; N = size(Ap, 1)
    a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)
    lattice_basis16(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(ACEWorkflow.Elasticity.reference_system(:Al; a=a), model))
    b_prime        = ForwardDiff.derivative(lattice_basis16, a_eq)
    b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis16, a), a_eq)
    say(@sprintf("a_eq = %.6f Å; %d parameters, %d observations", a_eq, length(lin_params), N))

    # pinned closed-form corrections, verbatim from al16/01
    λ  = 1.0 / N
    C  = Symmetric(P' * P .* λ .+ Ap' * Ap); Cf = cholesky(C)
    Ainv = Cf \ Matrix(Ap')
    x̃    = Cf \ (Ap' * Yw)
    lev  = vec(sum(Ap' .* Ainv; dims=1))
    g  = transpose(P) \ b_prime
    u  = Cf \ g; gu = dot(g, u); v = Ap * u
    x̃_c   = x̃ .- u .* (dot(g, x̃) / gu)
    lev_c = lev .- (v .^ 2) ./ gu
    r_c   = Yw .- Ap * x̃_c
    scale = r_c ./ lev_c
    PCx   = transpose(Ainv .* transpose(scale))
    PCx .-= (scale .* v ./ gu) * transpose(u)
    Ainv = nothing; GC.gc()
    keep_rows = lev_c .>= quantile(lev_c, 1 - LEV_PCT)
    pwise = PCx[keep_rows, :] / transpose(P)
    PCx = nothing; GC.gc()
    say(@sprintf("cloud: %d corrections (top %.0f%% constrained leverage); pin check max |b′·δθ| = %.2e",
                 size(pwise, 1), 100LEV_PCT, maximum(abs.(pwise * b_prime))))

    MODELDIR = model_dir("Al_16_4_6A_3_")
    bpf = firstfile(get(ENV, "BPCACHE", ""),
                    joinpath(MODELDIR, "results", "pinned_ensembles", "bandpath_4x4x4_aref.jls"),
                    joinpath(ROUT, "al16", "ensembles", "bandpath_4x4x4_aref.jls"))
    bpf === nothing && error("Al_16 band-path cache not found: run al16/01_pinned_ensembles.jl once (it writes " *
                             "bandpath_4x4x4_aref.jls) or set BPCACHE")
    bp = deserialize(bpf); keep_q = findall(bp.qnorm .>= 5e-2)
    say("band path: $bpf ($(length(bp.Bq)) q-points, $(length(keep_q)) after the near-Γ cut)")
    minw(θ) = minimum(minimum(sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz)
                      for ev in (eigvals(Hermitian(reshape(bp.Bq[iq] * θ, 3bp.Np, 3bp.Np))) for iq in keep_q))
    stages = [("b″·θ > 0",                                        θ -> dot(b_double_prime, θ) > 0),
              (@sprintf("min ω ≥ %.2f THz", ACCEPT_TOL),          θ -> minw(θ) >= ACCEPT_TOL)]

    eig_pops, bound_pops = hypercube(pwise)
    w = bound_pops[2, :] .- bound_pops[1, :]
    say(@sprintf("hypercube: %d directions (width mean %.4g, max %.4g); max |b′·eigvec| = %.2e",
                 size(eig_pops, 2), mean(w), maximum(w), maximum(abs.(transpose(eig_pops) * b_prime))))
    rec = let f = firstfile(joinpath(ROUT, "al16", "ensembles", "pinned_ensembles.jls"))
        f === nothing ? "n/a (al16/01 log)" : @sprintf("%.3f%%", 100deserialize(f).acceptance)
    end
    acc, n_prop = published_draw(eig_pops, bound_pops, lin_params, stages, N_MEMBERS)
    say(@sprintf("  published draw (seed 1234, %d members): %d proposals → %.3f%% acceptance   (recorded: %s)",
                 N_MEMBERS, n_prop, 100acc / n_prop, rec))
    Random.seed!(4321)
    t = @elapsed surv = funnel(eig_pops, bound_pops, lin_params, stages, N_PROP)
    report("Al_16", stages, surv, N_PROP, rec)
    say(@sprintf("  [%.1f s]", t))
end

write(joinpath(OUTDIR, "rejection_rate.txt"), join(LINES, "\n") * "\n")
println("\n→ $(joinpath(OUTDIR, "rejection_rate.txt"))")
