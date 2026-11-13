# Figure 2: RLS against constrained RLS at 200 K.  (a) phonon bands at the lattice constant each model reached in
# its NPT run; (b), (c) the radial distribution functions of the two runs, with the ideal FCC shells dashed.
#
# In:   results/fig2/bands_{rls,constrained}.csv, band_ticks.csv, {rls,constrained}/{a_200K,rdf_200K}.csv
#       (published/ if this run has none)
# Out:  results/fig2/fig_200K_rls_vs_constrained.{pdf,png}
# Run:  julia --project scripts/fig2_rls_200K/4_plot.jl
using PhysicsPriors, DelimitedFiles, Printf, CairoMakie

T_K  = 200
FIGW = 510.0
BLU, RED, ORA = RGBf(0.0, 0.447, 0.698), RGBf(0.80, 0.15, 0.15), RGBf(0.835, 0.369, 0.0)
TITLE, SUB, LAB, TICK, SMALL = 13, 13, 12, 11, 11

rls, con = map(("rls", "constrained")) do tag
    b  = readdlm(input("fig2", "bands_$tag.csv"), ','; skipstart=1)
    rg = readdlm(input("fig2", tag, "rdf_200K.csv"), ','; skipstart=1)
    (; x = Float64.(b[:, 1]), F = Matrix(Float64.(b[:, 2:end])'), r = Float64.(rg[:, 1]), g = Float64.(rg[:, 2]),
       a = readdlm(input("fig2", tag, "a_200K.csv"), ','; skipstart=1)[1])
end
ticks   = readdlm(input("fig2", "band_ticks.csv"), ','; skipstart=1)
x_ticks = Float64.(ticks[:, 2]); labels = String.(ticks[:, 1])
xc = con.x

fig = Figure(size=(FIGW, 0.66FIGW), figure_padding=(6, 10, 4, 6))
ax1 = Axis(fig[1:2, 1]; xlabel="Wave vector", ylabel="Frequency (THz)", title="Phonon dispersion at a($(T_K) K)",
           titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
           xticks=(x_ticks, labels), xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
lo, hi = extrema(vcat(vec(rls.F), vec(con.F))); pad = 0.06(hi - lo)
band!(ax1, [first(xc), last(xc)], [lo - pad, lo - pad], [0.0, 0.0]; color=(RED, 0.07))
for b in axes(rls.F, 1); lines!(ax1, rls.x, rls.F[b, :]; color=(RED, 0.95), linewidth=1.5); end
for b in axes(con.F, 1); lines!(ax1, xc, con.F[b, :]; color=BLU, linewidth=1.6); end
hlines!(ax1, [0.0]; color=:black, linestyle=:dash, linewidth=0.9)
vlines!(ax1, x_ticks; color=(:black, 0.22), linewidth=0.6)
xlims!(ax1, first(xc), last(xc)); ylims!(ax1, lo - pad, hi + 4pad)   # headroom for the legend
axislegend(ax1, [LineElement(color=RED, linewidth=2.2), LineElement(color=BLU, linewidth=2.2)],
           [@sprintf("RLS"), @sprintf("constrained RLS")];
           position=:rt, framevisible=true, labelsize=TICK, patchsize=(18, 2), padding=(5, 5, 3, 3), rowgap=1)
text!(ax1, 0.015, 0.985; text="(a)", space=:relative, align=(:left, :top), font=:bold, fontsize=TITLE)

shells(a, rmax) = filter(<(rmax), a .* sqrt.((1:16) ./ 2))     # ideal FCC shells r_n = a√(n/2)
rmax = min(maximum(rls.r), maximum(con.r))
axs = map(enumerate(((rls, RED, 0.70, "RLS", "(b)"), (con, BLU, 0.75, "Constrained RLS", "(c)")))) do (i, (x, col, α, ttl, tag))
    ax = Axis(fig[i, 2]; ylabel="g(r)", xlabel=i == 2 ? "r (Å)" : "", title=ttl, titlesize=SUB, titlecolor=col,
              xlabelsize=LAB, ylabelsize=LAB, xticklabelsize=TICK, yticklabelsize=TICK,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    barplot!(ax, x.r, x.g; width=step(range(x.r[1], x.r[end]; length=length(x.r))), color=(col, α), gap=0, strokewidth=0)
    vlines!(ax, shells(x.a, rmax); color=(ORA, 0.85), linestyle=:dash, linewidth=1.0)
    xlims!(ax, 0, rmax)
    i == 1 && (hidexdecorations!(ax; grid=false, ticks=false, minorticks=false);
               text!(ax, 0.985, 0.93; space=:relative, align=(:right, :top), fontsize=SMALL, color=ORA))
    text!(ax, 0.025, 0.95; text=tag, space=:relative, align=(:left, :top), font=:bold, fontsize=TITLE)
    ax
end
linkxaxes!(axs...)
colsize!(fig.layout, 1, Relative(0.54)); colgap!(fig.layout, 14); rowgap!(fig.layout, 6)

out = results("fig2", "fig_200K_rls_vs_constrained")
save("$out.pdf", fig); save("$out.png", fig; px_per_unit=4)
println("figure → $out.{pdf,png}")
