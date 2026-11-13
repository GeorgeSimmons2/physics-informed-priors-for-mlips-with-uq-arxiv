# Figure 1, Al_12, stage 4: the unconstrained ensemble, and the bands of both ensembles (top row of the figure).
#
# unconstrained  30 draws from the hypercube of the ordinary POPS cloud (same observations, no constraints).  Each
#                member has its own lattice constant, so each is relaxed and gets its own Hessian.
# constrained    the 30 members of stage 3.  They all sit at a_eq, so one undotted Hessian gives every band.
#
# In:   models/Al_12_4_6A_2_/, results/fig1/al12/theta_mean.csv, ensemble_constrained.csv
# Out:  results/fig1/al12/ensemble_unconstrained.csv, bands.jls
# Run:  julia --project -t 40 scripts/fig1_bands/al12_4_bands.jl
using PhysicsPriors, Random, DelimitedFiles, Serialization, Printf
Random.seed!(1234)

N_cell, N_per_seg = 4, [20, 20, 20, 20, 60]
lev_percentile    = 0.5
n_members         = 30
unstable_THz      = -0.05                        # a member is called unstable below this (recorded with the bands)

m    = load_model(AL12)
prob = regression_problem(m)
a_eq = relax_lattice_constant(m.model)
bp   = bandpath_basis(m, a_eq, N_cell; N_per_seg)

# unconstrained: POPS corrections → hypercube → sample
eigvecs, bounds = hypercube(corrections(prob.Ap, prob.Yw, prob.P; leverage_percentile=lev_percentile))
unconstrained, _ = sample_hypercube(eigvecs, bounds, m.lin_params; number_of_committee_members=n_members)
writedlm(results("fig1", "al12", "ensemble_unconstrained.csv"), unconstrained', ',')
members_u = [unconstrained[:, i] for i in 1:n_members]

constrained = readdlm(input("fig1", "al12", "ensemble_constrained.csv"), ',')
members_c   = [collect(Float64, constrained[i, :]) for i in 1:n_members]
θ_mean      = vec(readdlm(input("fig1", "al12", "theta_mean.csv"), ','))

# bands: each unconstrained member at its own relaxed lattice constant; the constrained ones on the a_eq basis
bands_u  = map_with_model_copies(m.model, members_u) do model, θ
    native_bands(model, θ, relax_or_fallback(model, θ, a_eq); N_cell, N_per_seg)
end
centre_u = native_bands(deepcopy(m.model), m.lin_params, a_eq; N_cell, N_per_seg)      # the RLS model
bands_c  = [bands_record(θ, bp, a_eq) for θ in members_c]
centre_c = bands_record(θ_mean, bp, a_eq)                                              # the constrained mean

for (tag, b) in (("unconstrained", bands_u), ("constrained", bands_c))
    @printf("%-14s min ω ∈ [%+.3f, %+.3f] THz, %d of %d unstable\n", tag, extrema(x.minω for x in b)...,
            count(x -> x.minω < unstable_THz, b), length(b))
end

allF = vcat([b.F for b in bands_u], [b.F for b in bands_c], [centre_u.F, centre_c.F])
ylo = minimum(minimum.(allF)); yhi = maximum(maximum.(allF)); pad = 0.06*(yhi-ylo)
ylim = (min(ylo-pad, -0.4), yhi+pad)
serialize(results("fig1", "al12", "bands.jls"),
          (; res = [bands_u, bands_c], cen = [centre_u, centre_c], mean_b = centre_c,
             tags = ["unconstrained", "constrained"], ylim,
             N_CELL = N_cell, N_PER_SEG = N_per_seg, UNSTABLE = unstable_THz, n_params = length(m.lin_params), a_eq,
             model_name = m.name))
