# Figure 3, stage 4: committees of 100 members from the two clouds.
#
# constrained    the hypercube of the constrained cloud about θc, rejection-sampled on the same windows as the fit
#                (C11, C12, C44 within ±0.1 %, |P(a0)| ≤ 0.001 GPa)
# unconstrained  the hypercube of the unconstrained cloud about θ0, sampled without rejection
# Both from seed 1234.
#
# In:   results/fig3/clouds.jls
# Out:  results/fig3/committee_{constrained,unconstrained}.csv      100 members each, one per row
# Run:  OMP_NUM_THREADS=16 julia --project scripts/fig3_mace_elastic/4_committees.jl
using PhysicsPriors, Random, Serialization, DelimitedFiles

n_members = 100
c = deserialize(input("fig3", "clouds.jls"))

eigvecs, bounds = hypercube(Matrix((c.Θ_con .- c.θc)'))
Random.seed!(1234)
constrained, _ = rejection_sample_hypercube(eigvecs, bounds, c.θc, c.R, (c.lo, c.hi);
                                            number_of_committee_members=n_members, max_attempts=5_000_000)

eigvecs_u, bounds_u = hypercube(Matrix((c.Θ_unc .- c.θ0)'))
Random.seed!(1234)
unconstrained, _ = sample_hypercube(eigvecs_u, bounds_u, c.θ0; number_of_committee_members=n_members)

writedlm(results("fig3", "committee_constrained.csv"), constrained', ',')
writedlm(results("fig3", "committee_unconstrained.csv"), unconstrained', ',')
