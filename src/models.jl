# The two ACE models of the paper.  A fitted model is a directory models/<name>_/ holding
#   <name>.json       the ACE basis (ACEpotentials)
#   lin_params.csv    the fitted parameters θ (ridge least squares, RLS)
#   A.csv, Y.csv, W.csv, P.csv   design matrix, targets, weights and the smoothness prior of the fit

"Al, total degree 12, correlation order 2: 91 parameters."
const AL12 = (name = "Al_12_4_6A_2", totaldegree = 12, order = 2)
"Al, total degree 16, correlation order 3: 684 parameters."
const AL16 = (name = "Al_16_4_6A_3", totaldegree = 16, order = 3)
const SMOOTHNESS, RCUT = 4, 6.0

"The directory of a fitted model: `models/<name>_/`, or under `\$ACE_MODELS` (models you have refitted yourself)."
model_dir(spec) = abspath(joinpath(get(ENV, "ACE_MODELS", joinpath(root(), "models")), spec.name * "_"))

"""
    load_model(spec; design=true) -> (; model, lin_params, name, dir[, A, Y, P, W])

A fitted model with θ set.  `design=false` skips the design matrix (A.csv is 250 MB for AL12, 1.9 GB for AL16).
"""
function load_model(spec; design=true)
    dir = model_dir(spec)
    model, _ = ACEpotentials.load_model(joinpath(dir, "$(spec.name).json"))
    lin_params = vec(readdlm(joinpath(dir, "lin_params.csv"), ','))
    set_linear_parameters!(model, lin_params)
    design || return (; model, lin_params, name=spec.name, dir)
    A  = readdlm(joinpath(dir, "A.csv"), ',')
    Y  = vec(readdlm(joinpath(dir, "Y.csv"), ','))
    P  = readdlm(joinpath(dir, "P.csv"), ',')
    W  = vec(readdlm(joinpath(dir, "W.csv"), ','))
    return (; model, A, Y, P, W, lin_params, name=spec.name, dir)
end

"""
    regression_problem(m) -> (; Ap, Yw, P, λ)

The weighted, preconditioned least-squares problem the model was fitted on, in the variable x = Pθ:
minimise ‖Ap x − Yw‖² + λ‖x‖², with `Ap = W A P⁻¹`, `Yw = W Y` and `λ = 1/(number of observations)`.
"""
function regression_problem(m)
    P = m.P; Ap = Diagonal(m.W)*m.A/P; Yw = m.W.*m.Y; λ = 1.0/size(Ap,1)
    return (; Ap, Yw, P, λ)
end

"""
    fit_model(spec; training_xyz, dir)

Fit an ACE model from DFT data and write it to `dir`: energies weighted 30, forces and virials 1, ACEpotentials'
smoothness prior P, and θ = P⁻¹ (ÃᵀÃ + PᵀP/N)⁻¹ Ãᵀỹ.  Assembly of the design matrix uses the Distributed workers.
"""
function fit_model(spec; training_xyz, dir)
    mkpath(dir)
    training_configs = ExtXYZ.load(training_xyz)
    model = ace1_model(elements=[:Al], totaldegree=spec.totaldegree, order=spec.order, rcut=RCUT)
    data = ACEpotentials.make_atoms_data(training_configs, model;
                energy_key = :dft_energy,
                force_key  = :dft_forces,
                virial_key = :dft_virials,
                weights    = Dict("default" => Dict("E"=>30., "F"=>1., "V"=>1.)))
    P = ACEpotentials._make_prior(model, SMOOTHNESS, nothing)
    A, Y, W = ACEfit.assemble(data, model)
    Ap       = Diagonal(W) * A / P
    Yw       = W .* Y
    A_reg    = Ap'*Ap .+ P'*P ./ size(Ap, 1)
    lin_params = P \ (A_reg \ (Ap' * Yw))
    set_linear_parameters!(model, lin_params)
    ACEpotentials.save_model(model, joinpath(dir, "$(spec.name).json"))
    writedlm(joinpath(dir, "A.csv"),          A,          ',')
    writedlm(joinpath(dir, "Y.csv"),          Y,          ',')
    writedlm(joinpath(dir, "P.csv"),          P,          ',')
    writedlm(joinpath(dir, "W.csv"),          W,          ',')
    writedlm(joinpath(dir, "lin_params.csv"), lin_params, ',')
    @info "fitted $(spec.name): $(length(lin_params)) parameters, $(size(A, 1)) observations → $dir"
end
