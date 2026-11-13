# Figure 1: phonon bands of unconstrained (left) and constrained (right) POPS ensembles, for the 91-parameter
# model (top) and the 684-parameter model (bottom).  Mean model blue, every ensemble member grey.
#
# In:   results/fig1/al12/bands.jls, results/fig1/al16/bands.jls      (published/ if this run has none)
# Out:  results/fig1/bands_four_panel.{pdf,png}
# Run:  julia --project scripts/fig1_bands/plot.jl
using PhysicsPriors, Serialization, Statistics, Printf, CairoMakie

FIGW, ASPECT = 510.0, 0.78               # points: the width the figure is displayed at in the paper
BLU  = RGBf(0.0, 0.447, 0.698)
GRY  = RGBAf(0.45, 0.45, 0.45, 0.38)
TITLE, LAB, TICK = 13, 12, 11

d12 = deserialize(input("fig1", "al12", "bands.jls"))
d16 = deserialize(input("fig1", "al16", "bands.jls"))
rows = [(d = d12, label = "Al_12_4_6A_2_\n(91 params)"),
        (d = d16, label = "Al_16_4_6A_3_\n(684 params)")]

# The central model of each panel.  Al_12's two panels have different centres (the RLS model at its own lattice
# constant on the left, the constrained mean on the right); Al_16 pins both ensembles to one mean model.
centre(d, k) = haskey(d, :cen) ? d.cen[k] : d.mean_b

for (i, r) in enumerate(rows), (k, tag) in enumerate(r.d.tags)
    w = [b.minω for b in r.d.res[k]]
    @printf("row %d %-14s %2d members, %2d below %.2f THz, min ω ∈ [%+.4f, %+.4f]\n",
            i, tag, length(w), count(<(r.d.UNSTABLE), w), r.d.UNSTABLE, minimum(w), maximum(w))
end

# x: the two models relax to different lattice constants, so their paths differ in length by ~0.3 %.  Each row's
# path coordinate is mapped piecewise-linearly onto the mean of the two tick vectors, so the ticks line up.
lbl = rows[1].d.mean_b.labels
tickmats = [r.d.mean_b.x_ticks for r in rows]
xt = mean(tickmats)
function remap(x, from, to)
    y = similar(x)
    for (i, xi) in enumerate(x)
        j = clamp(searchsortedlast(from, xi), 1, length(from) - 1)
        t = (xi - from[j]) / (from[j+1] - from[j])
        y[i] = to[j] + t * (to[j+1] - to[j])
    end
    return y
end
xr = [remap(r.d.mean_b.x_vals, r.d.mean_b.x_ticks, xt) for r in rows]

# y: shared within a row (the comparison the figure makes), independent between rows
ylims = [r.d.ylim for r in rows]

fig = Figure(size = (FIGW, ASPECT * FIGW), figure_padding = (6, 10, 4, 6))
for i in 1:2, k in 1:2
    r = rows[i]; bs = r.d.res[k]; Np = r.d.mean_b.Np
    bottom = i == 2
    ax = Axis(fig[i + 1, k];
              xlabel = bottom ? "Wave vector" : "",
              ylabel = k == 1 ? "Frequency (THz)" : "",
              title  = i == 1 ? r.d.tags[k] : "", titlesize = TITLE,
              xlabelsize = LAB, ylabelsize = LAB,
              xticklabelsize = TICK, yticklabelsize = TICK,
              xticks = (xt, lbl), xgridvisible = false, ygridvisible = false,
              xtickalign = 1, ytickalign = 1, xticksize = 4, yticksize = 4)
    for b in bs, br in 1:3Np
        lines!(ax, xr[i], b.F[br, :]; color = GRY, linewidth = 0.8)
    end
    c = centre(r.d, k)
    for br in 1:3Np; lines!(ax, xr[i], c.F[br, :]; color = BLU, linewidth = 1.8); end
    hlines!(ax, [0.0]; color = :black, linestyle = :dash, linewidth = 1.0)
    vlines!(ax, xt; color = (:black, 0.22), linewidth = 0.7)
    xlims!(ax, first(xt), last(xt)); ylims!(ax, ylims[i]...)
    k == 1 || hideydecorations!(ax; grid = false, ticks = false, minorticks = false)
    bottom  || hidexdecorations!(ax; grid = false, ticks = false, minorticks = false)
end
for i in 1:2
    Label(fig[i + 1, 0], rows[i].label; rotation = π/2, fontsize = LAB, tellheight = false)
end
Legend(fig[1, 1:2],
       [LineElement(color = BLU, linewidth = 2.2),
        LineElement(color = RGBf(0.45, 0.45, 0.45), linewidth = 2.2),
        LineElement(color = :black, linestyle = :dash, linewidth = 1.6)],
       ["mean model", "ensemble member", "ω = 0"];
       orientation = :horizontal, tellwidth = false, framevisible = false,
       labelsize = TICK, padding = (2, 2, 0, 0), colgap = 16)
rowgap!(fig.layout, 1, 2); rowgap!(fig.layout, 2, 8)
colgap!(fig.layout, 1, 4); colgap!(fig.layout, 2, 16)

out = results("fig1", "bands_four_panel")
save("$out.pdf", fig); save("$out.png", fig; px_per_unit = 4)
println("figure → $out.{pdf,png}")
