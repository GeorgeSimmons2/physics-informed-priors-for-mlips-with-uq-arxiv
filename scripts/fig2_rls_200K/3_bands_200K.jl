# Figure 2, stage 3: the phonon bands of both models at the lattice constant each reached in its 200 K run.
#
# In:   models/Al_12_4_6A_2_/, results/fig2/theta_constrained.csv, results/fig2/{rls,constrained}/a_200K.csv
# Out:  results/fig2/bands_{rls,constrained}.csv, band_ticks.csv   (and two 4×4×4 undotted Hessians, cached)
# Run:  julia --project -t 8 scripts/fig2_rls_200K/3_bands_200K.jl
using PhysicsPriors, DelimitedFiles, Printf

N_cell, N_per_seg = 4, [20, 20, 20, 20, 60]

m = load_model(AL12; design=false)
runs = map(("rls", "constrained")) do tag
    θ  = tag == "rls" ? m.lin_params : vec(readdlm(input("fig2", "theta_constrained.csv"), ','))
    a  = readdlm(input("fig2", tag, "a_200K.csv"), ','; skipstart=1)[1]
    bp = bandpath_basis(m, a, N_cell; N_per_seg)
    @printf("%-12s a(200 K) = %.5f Å, min ω = %+.3f THz\n", tag, a, min_freq_stable(θ, bp))
    (; tag, bp, F = bands(θ, bp))
end
rls, con = runs

# One path coordinate for both: the constrained model's.  |q| scales as 1/a on every segment, so the RLS path is
# rescaled onto it exactly.
xc = con.bp.x_vals
xplot(x) = x.bp.x_vals .* (last(xc) / last(x.bp.x_vals))
for x in runs
    writedlm(results("fig2", "bands_$(x.tag).csv"), vcat(["x_plot" "branch1_THz" "branch2_THz" "branch3_THz"], hcat(xplot(x), x.F')), ',')
end
writedlm(results("fig2", "band_ticks.csv"), vcat(["label" "x_plot"], hcat(con.bp.labels, con.bp.x_ticks)), ',')
