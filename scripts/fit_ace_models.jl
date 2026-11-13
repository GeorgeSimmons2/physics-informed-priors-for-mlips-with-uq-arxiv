# Refit the two ACE models from the DFT training set (the fitted models are already in models/).
#
#   Al_12_4_6A_2   total degree 12, correlation order 2, cutoff 6 Å    91 parameters
#   Al_16_4_6A_3   total degree 16, correlation order 3, cutoff 6 Å   684 parameters
#
# Both on all 7,893 frames of data/Al/manual_df_train_Al.extxyz (146,958 energy, force and virial rows).  The fit
# is deterministic: it reproduces the published parameters up to floating-point summation order.
#
# In:   data/Al/manual_df_train_Al.extxyz
# Out:  models_refit/<name>_/      (use them with ACE_MODELS=models_refit; models/ is never overwritten)
# Run:  NPROCS=16 julia --project scripts/fit_ace_models.jl
using Distributed
addprocs(parse(Int, get(ENV, "NPROCS", "8")); exeflags="--project=$(Base.active_project())")
@everywhere using PhysicsPriors

training_xyz = joinpath(root(), "data", "Al", "manual_df_train_Al.extxyz")
for spec in (AL12, AL16)
    fit_model(spec; training_xyz, dir=joinpath(root(), "models_refit", spec.name * "_"))
end
