# Lattice constant and elasticity of FCC Al as rows that are linear in θ.
#
# The energy of a cell is E = B·θ with B the summed basis.  Differentiating B with respect to the lattice constant
# or a strain gives rows whose dot product with θ is the pressure, the bulk curvature or an elastic constant, so
# "the lattice constant is a_eq" and "the crystal is Born stable" are linear constraints on θ.

function Voigt_strain_to_3x3(strain_vector::SVector{6,T}) where {T}
    s1, s2, s3, s4, s5, s6 = strain_vector
    return @SMatrix [
        s1       0.5*s6   0.5*s5
        0.5*s6   s2       0.5*s4
        0.5*s5   0.5*s4   s3
    ]
end

function lattice_matrix(cell_vectors)
    a, b, c = cell_vectors
    return SMatrix{3,3,eltype(a)}(hcat(a, b, c))
end
lattice_tuple(L) = (L[:,1], L[:,2], L[:,3])

"The primitive cell of `element` at lattice constant `a` (Å; nothing → AtomsBuilder's default)."
function reference_system(element; a=nothing)
    if isnothing(a)
        return bulk(element)
    else
        a_u = a isa Unitful.Quantity ? a : a * u"Å"
        return bulk(element; a=a_u, T=typeof(ustrip(a_u)))
    end
end

function fractional_positions_unitless(sys)
    L0 = SMatrix{3,3,Float64}(ustrip.(lattice_matrix(sys.cell.cell_vectors)))
    Linv = inv(L0)
    fracs = Vector{SVector{3,Float64}}(undef, length(sys))
    for i in 1:length(sys)
        r = SVector{3,Float64}(ustrip.(sys[i].position))
        fracs[i] = Linv * r
    end
    return fracs
end

function rebuild_periodic_system_unitful(sys; Lnew_unitless, fracs)
    Lnew = Lnew_unitless .* u"Å"
    atoms_new = Vector{Atom}(undef, length(sys))
    for i in 1:length(sys)
        at = sys[i]
        rnew = Lnew * fracs[i]
        atoms_new[i] = Atom(at.species, collect(rnew), missing)
    end
    return periodic_system(atoms_new, lattice_tuple(Lnew))
end

"The energy basis of the reference cell under the Voigt strain ε (differentiable in ε)."
function strained_cell_design(ε::SVector{6,T}; model, element, a=nothing) where {T}
    sys0 = reference_system(element; a=a)
    ϵ = Voigt_strain_to_3x3(ε)
    F = ϵ + one(ϵ)
    L0 = SMatrix{3,3,T}(ustrip.(lattice_matrix(sys0.cell.cell_vectors)))
    L1 = F * L0
    fracs = fractional_positions_unitless(sys0)
    sys1 = rebuild_periodic_system_unitful(sys0; Lnew_unitless=L1, fracs=fracs)
    return ACEpotentials.Models.potential_energy_basis(sys1, model)
end

"∂²B/∂ε² at zero strain: a 6 × 6 × n_params array, the strain Hessian of every basis function."
function elastic_hessian_basis(model; element=:Al, a=nothing, ε0=nothing)
    ε_ref = isnothing(ε0) ? @SVector(zeros(6)) : ε0
    f(ε) = strained_cell_design(ε; model=model, element=element, a=a)
    nparams = length(f(ε_ref))
    H_6x6xd = Array{Float64}(undef, 6, 6, nparams)
    for k in 1:nparams
        g(ε) = ustrip.(f(ε)[k])
        H_6x6xd[:, :, k] = ForwardDiff.hessian(g, ε_ref)
    end
    return H_6x6xd
end

"""
    relax_lattice_constant(model, element=:Al) -> Float64

The equilibrium lattice constant (Å) of the model at its current parameters, from a variable-cell energy
minimisation of the primitive cell.
"""
function relax_lattice_constant(model, element::Symbol=:Al)
    sys = reference_system(element)
    res = minimize_energy!(sys, model; variablecell=true)
    optsys = res.system
    return ustrip(u"Å", optsys.cell.cell_vectors[1][2] * 2)
end

"""
    relax_or_fallback(model, θ, a_ref) -> Float64

The relaxed lattice constant of θ, or `a_ref` when the relaxation fails or lands outside 0.9–1.1 `a_ref` (an
unconstrained ensemble member need not have a sensible minimum).  Sets the model's parameters to θ.
"""
function relax_or_fallback(model, θ, a_ref)
    set_linear_parameters!(model, θ)
    try
        aa = relax_lattice_constant(model)
        (0.9a_ref < aa < 1.1a_ref) ? aa :
            (@warn "relaxed a = $aa Å implausible; falling back to a_ref"; a_ref)
    catch e
        @warn "relax failed ($e); falling back to a_ref"; a_ref
    end
end

"The energy basis of the reference cell at lattice constant `a`, in eV (differentiable in `a`)."
lattice_basis(model, element, a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(reference_system(element; a=a), model))

"""
    lattice_rows(model, a_eq; element=:Al) -> (; b_prime, b_double_prime)

First and second derivative of the cell's energy basis with respect to the lattice constant at `a_eq`.
`b_prime·θ = 0` pins the lattice constant of θ to `a_eq`; `b_double_prime·θ > 0` makes it a minimum.
"""
function lattice_rows(model, a_eq; element=:Al)
    lb(a) = lattice_basis(model, element, a)
    b_prime        = ForwardDiff.derivative(lb, a_eq)
    b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lb, a), a_eq)
    return (; b_prime, b_double_prime)
end

"""
    cubic_elastic_rows(model, a_eq; element=:Al) -> (; c11, c12, c44, eV_to_GPa)

The rows of the strain Hessian that give C11, C12 and C44, in eV, and the factor that turns them into GPa for the
reference cell at `a_eq`.
"""
function cubic_elastic_rows(model, a_eq; element=:Al)
    sys0 = reference_system(element; a=a_eq)
    L0   = ustrip.(lattice_matrix(sys0.cell.cell_vectors)); eV_to_GPa = 160.2176621/abs(det(L0))
    H_el = elastic_hessian_basis(model; element=element, a=a_eq)
    c11 = reshape(H_el,36,:)[1,:]; c12 = reshape(H_el,36,:)[7,:]; c44 = reshape(H_el,36,:)[22,:]
    return (; c11, c12, c44, eV_to_GPa)
end

"The cubic elastic constants (C11, C12, C44) of θ in GPa."
elastic_constants(el, θ) = (dot(el.c11,θ)*el.eV_to_GPa, dot(el.c12,θ)*el.eV_to_GPa, dot(el.c44,θ)*el.eV_to_GPa)

"""
    lattice_shift(θ, lat; θ_ref=nothing) -> Float64

Newton estimate of how far the lattice constant of θ is from the `a_eq` the rows `lat` were built at:
Δa = −b′·(θ − θ_ref) / b″·θ (θ_ref omitted: −b′·θ / b″·θ).
"""
lattice_shift(θ, lat; θ_ref=nothing) =
    θ_ref === nothing ? -dot(lat.b_prime, θ)/dot(lat.b_double_prime, θ) :
                        -dot(lat.b_prime, θ .- θ_ref) / dot(lat.b_double_prime, θ)

# ── Born stability ───────────────────────────────────────────────────────────────────────────────────────────────

"""
    born_constraints(el, lat) -> (; rows, lower)

The Born stability conditions of a cubic crystal as `rows*θ .>= lower`: C44, C11 − C12, C11 + 2 C12 (eV rows) and
the curvature b″ of the energy in the lattice constant, with `lower = [0.1, 1.0, 0.1, 1e-9]`.
"""
function born_constraints(el, lat)
    rows  = vcat(el.c44', (el.c11.-el.c12)', (el.c11.+2 .*el.c12)', lat.b_double_prime')
    lower = [0.1, 1.0, 0.1, 1e-9]
    return (; rows, lower)
end

"Whether θ satisfies every Born condition."
is_born_stable(born, θ) = all(born.lower .<= born.rows*θ)

"Whether the lattice constant of θ stays close to that of `θ_mean`: |b′·(θ − θ_mean) / (θ_mean·b″)| ≤ tol."
lattice_drift_ok(θ, lat, θ_mean; tol=0.1) = abs(dot(lat.b_prime, θ .- θ_mean)/dot(θ_mean, lat.b_double_prime)) <= tol

"Whether the energy of θ is a minimum in the lattice constant: b″·θ > 0."
positive_curvature(θ, lat) = dot(lat.b_double_prime, θ) > 0
