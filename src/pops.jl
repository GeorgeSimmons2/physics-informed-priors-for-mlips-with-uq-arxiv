# POPS (pointwise optimal parameter sets): for every training observation, the smallest change of the fitted
# parameters that reproduces that observation exactly.  The cloud of these corrections is boxed in its principal
# axes (the hypercube) and ensembles are drawn uniformly from the box, optionally keeping only physical draws.

"""
    corrections(X, Y, Gamma; leverage_percentile=0.5, lambda=1/size(X,1), coeffs=nothing) -> N × n_params

The POPS corrections δθ of the ridge fit of `Y` on `X` (prior `Gamma`), one row per observation whose leverage is
at or above the `leverage_percentile` quantile.
"""
function corrections(X::AbstractMatrix{Float64}, Y::Vector{Float64}, Gamma::AbstractMatrix{Float64}; leverage_percentile::Float64 = 0.5, lambda::Float64 = 1.0 / size(X,1), coeffs=nothing)
    C      = (Gamma' * Gamma .* lambda .+ X' * X)
    A      = C \ X'
    leverage = vec(sum(X .* A'; dims=2))
    if (coeffs == nothing)
        coeffs = C \ (X' * Y)
    end
    errors = Y .- (X * coeffs)
    leverage_threshold = quantile(leverage, leverage_percentile)
    mask = leverage .>= leverage_threshold
    pointwise_corrections = A[:,mask]'
    pointwise_corrections = pointwise_corrections .* (errors[mask] ./ leverage[mask])
    pointwise_corrections = Gamma \ pointwise_corrections'
    return pointwise_corrections'
end

"""
    hypercube(pointwise_corrections) -> (eigvecs, bounds)

The principal axes of a cloud of corrections (eigenvalues above 1e-8 of the largest) and, per axis, the lowest
and highest projection of the cloud (`bounds` is 2 × n_axes).
"""
function hypercube(pointwise_corrections::AbstractMatrix{Float64}; percentile_clipping::Float64 = 0.0)
    eig = eigen(Symmetric(pointwise_corrections' * pointwise_corrections))
    eigvals = eig.values
    eigvecs = eig.vectors
    mask = eigvals .> maximum(eigvals) * 1e-8
    eigvecs = eigvecs[:, mask]
    eigvals = eigvals[mask]
    projections = eigvecs
    projected = pointwise_corrections * projections
    lower = [quantile(projected[:, j], percentile_clipping / 100) for j in 1:size(projected, 2)]
    upper = [quantile(projected[:, j], 1.0 - percentile_clipping / 100) for j in 1:size(projected, 2)]
    bounds = vcat(lower', upper')
    return eigvecs, bounds
end

"""
    sample_hypercube(eigvecs, bounds, coeffs; number_of_committee_members=50) -> (committee, δθ)

Members drawn uniformly from the hypercube about `coeffs`, one per column.  One `rand(N, d)` matrix is drawn, so
member i depends on N.
"""
function sample_hypercube(eigvecs::AbstractMatrix{Float64}, bounds::AbstractMatrix{Float64}, coeffs::Vector{Float64}; number_of_committee_members::Int64 = 50)
    lower, upper = bounds[1, :], bounds[2, :]
    U = rand(Float64, (number_of_committee_members, size(lower, 1)))
    committee = eigvecs * (lower[:, :]' .+ (upper .- lower)[:,:]' .* U)'
    δθ        = committee * committee' ./ size(committee, 2)
    committee = coeffs[:,:] .+ committee
    return committee, δθ
end

"""
    rejection_sample_hypercube(eigvecs, bounds, coeffs, is_feasible; number_of_committee_members=50, max_attempts)

As `sample_hypercube`, keeping only the draws θ for which `is_feasible(θ)` holds.  One `rand(d)` vector is drawn
per proposal, so the first n members of a longer run are the members of a run of n.
"""
function rejection_sample_hypercube(eigvecs::AbstractMatrix{Float64}, bounds::AbstractMatrix{Float64}, coeffs::Vector{Float64},
                                    is_feasible::Function;
                                    number_of_committee_members::Int64 = 50,
                                    max_attempts::Int64 = 10_000 * number_of_committee_members)
    lower, upper = bounds[1, :], bounds[2, :]
    perturbations = zeros(Float64, length(coeffs), number_of_committee_members)
    accepted = 0
    attempts = 0
    while accepted < number_of_committee_members
        if attempts >= max_attempts
            error("rejection_sample_hypercube: accepted only $accepted / " *
                  "$number_of_committee_members members after $attempts proposals. " *
                  "The feasible fraction of the hypercube is very small — check that " *
                  "the mean coefficients satisfy the constraints, or raise max_attempts.")
        end
        attempts += 1
        u = rand(Float64, size(lower, 1))
        δ = eigvecs * (lower .+ (upper .- lower) .* u)
        if is_feasible(coeffs .+ δ)
            accepted += 1
            perturbations[:, accepted] = δ
        end
    end
    @info "rejection_sample_hypercube: accepted $number_of_committee_members / $attempts proposals " *
          "($(round(100 * number_of_committee_members / attempts; digits=2))% acceptance)"
    δθ        = perturbations * perturbations' ./ number_of_committee_members
    committee = coeffs[:, :] .+ perturbations
    return committee, δθ
end

"Rejection sampling with linear constraints `l ≤ constraint_matrix·θ ≤ u` (`constraint_bounds = (l, u)`)."
function rejection_sample_hypercube(eigvecs::AbstractMatrix{Float64}, bounds::AbstractMatrix{Float64}, coeffs::Vector{Float64},
                                    constraint_matrix::AbstractMatrix, constraint_bounds::Tuple;
                                    kwargs...)
    lower_constraint, upper_constraint = constraint_bounds
    if any(lower_constraint .== upper_constraint)
        throw(ArgumentError(
            "constraint rows with equal lower and upper bounds (equality constraints) " *
            "can never be satisfied by rejection sampling from a continuous proposal; " *
            "pass only the inequality rows."))
    end
    is_feasible = c -> begin
        Ac = constraint_matrix * c
        all(lower_constraint .<= Ac) && all(Ac .<= upper_constraint)
    end
    return rejection_sample_hypercube(eigvecs, bounds, coeffs, is_feasible; kwargs...)
end

"""
    top_leverage_rows(prob, fraction) -> Vector{Int}

The observations in the top `fraction` by leverage h_i = a_i (ÃᵀÃ + λPᵀP)⁻¹ a_iᵀ: the rows whose corrections
make up a POPS cloud.
"""
function top_leverage_rows(prob, fraction)
    (; Ap, P, λ) = prob
    C   = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C)
    AtX = Cf \ Matrix(Ap')
    lev = vec(sum(Ap' .* AtX; dims=1))
    thresh = quantile(lev, 1 - fraction)
    return findall(>=(thresh), lev)
end

"""
    pinned_corrections(prob, b_prime; fraction, pin_tol) -> N × n_params

POPS corrections that keep the lattice constant fixed: each δθ reproduces its observation subject to b′·δθ = 0,
a closed-form rank-one update of the unconstrained correction,

    g = P′⁻¹ b′,  u = C⁻¹ g,  v = Ã u,   δx_i = (C⁻¹ a_i − u v_i / g·u) r_i / h_i

with r and h the residual and leverage of the pinned fit.  Returns the corrections of the top `fraction` of
observations by pinned leverage, in θ-space; errors if the pin fails to hold to `pin_tol`.
"""
function pinned_corrections(prob, b_prime; fraction, pin_tol)
    (; Ap, Yw, P, λ) = prob
    N  = size(Ap, 1)
    C  = Symmetric(P'*P .* λ .+ Ap'*Ap); Cf = cholesky(C)
    Ainv = Cf \ Matrix(Ap')
    x̃    = Cf \ (Ap' * Yw)
    lev  = vec(sum(Ap' .* Ainv; dims=1))
    g  = transpose(P) \ b_prime
    u  = Cf \ g; gu = dot(g, u); v = Ap * u
    abs(gu) > 1e-30 || error("g′u ≈ 0 — the pin direction lies in the null space of C⁻¹")
    x̃_c   = x̃ .- u .* (dot(g, x̃)/gu)
    lev_c = lev .- (v .^ 2) ./ gu
    r_c   = Yw .- Ap * x̃_c
    scale = r_c ./ lev_c
    PCx   = transpose(Ainv .* transpose(scale))
    PCx .-= (scale .* v ./ gu) * transpose(u)
    Ainv = nothing; GC.gc()
    @printf("pin check (all %d corrections): max |b′·δ| = %.3e (tolerance %.0e)\n", N, maximum(abs.(PCx * g)), pin_tol)
    maximum(abs.(PCx * g)) < pin_tol ||
        error("the pin did not hold on the forest — everything downstream is invalid")
    keep_rows = lev_c .>= quantile(lev_c, 1 - fraction)
    pwise = PCx[keep_rows, :] / transpose(P)
    @printf("cloud: %d of %d corrections (top %.0f%% pinned leverage); max |b′·δθ| = %.3e\n",
            count(keep_rows), N, 100fraction, maximum(abs.(pwise * b_prime)))
    return pwise
end
