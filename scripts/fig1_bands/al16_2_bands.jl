# Figure 1, Al_16, stage 2: the bands of both ensembles (bottom row of the figure).
#
# Every member is relaxed to its own lattice constant and its bands come from its own Hessian, independently of
# the undotted Hessian that screened the constrained draws in stage 1.
#
# In:   models/Al_16_4_6A_3_/, results/fig1/al16/ensemble_{unconstrained,constrained}.csv
# Out:  results/fig1/al16/bands.jls
# Run:  julia --project -t 40 scripts/fig1_bands/al16_2_bands.jl
using PhysicsPriors, DelimitedFiles, Serialization, Printf

N_cell, N_per_seg = 4, [20, 20, 20, 20, 60]
unstable_THz      = -0.05
tags              = ["unconstrained", "constrained"]

m = load_model(AL16; design=false)
ensembles = map(tags) do tag
    M = readdlm(input("fig1", "al16", "ensemble_$tag.csv"), ',')
    [collect(Float64, M[i, :]) for i in 1:size(M, 1)]
end

mean_b = native_bands(deepcopy(m.model), m.lin_params; N_cell, N_per_seg)
res = [map_with_model_copies((model, θ) -> native_bands(model, θ; N_cell, N_per_seg), m.model, members) for members in ensembles]

for (tag, b) in zip(tags, res)
    @printf("%-14s min ω ∈ [%+.4f, %+.4f] THz, %d of %d unstable, a ∈ [%.5f, %.5f] Å\n", tag, extrema(x.minω for x in b)...,
            count(x -> x.minω < unstable_THz, b), length(b), extrema(x.a for x in b)...)
end

allF = vcat([b.F for r in res for b in r], [mean_b.F])
lo = minimum(minimum.(allF)); hi = maximum(maximum.(allF)); pad = 0.06*(hi-lo)
ylim = (min(lo-pad, -0.5), hi+pad)
serialize(results("fig1", "al16", "bands.jls"),
          (; res, mean_b, ylim, N_CELL = N_cell, N_PER_SEG = N_per_seg, UNSTABLE = unstable_THz, tags))
