# Molecular dynamics at constant pressure and temperature (Molly), and the radial distribution function of its
# trajectory.  Used for Figure 2: the same 200 K run with the RLS parameters and with the constrained ones.

"""
    run_npt(model, θ; a0, T_K, seed, supercell=(4, 4, 4), n_equil=10_000, n_prod=20_000, log_every=50)

Langevin dynamics (1 fs, friction 0.01 fs⁻¹) with a Monte Carlo barostat at zero pressure, from the perfect FCC
supercell at lattice constant `a0` with random velocities at `T_K`.  Returns the Molly system; its loggers hold
temperature, coordinates, volume and potential energy every `log_every` steps.
"""
function run_npt(model, θ; a0, T_K, seed, supercell=(4, 4, 4), n_equil=10_000, n_prod=20_000, log_every=50)
    pressure, dt, friction = 0.0u"GPa", 1.0u"fs", 0.01u"fs^-1"
    Random.seed!(seed)
    T = T_K * u"K"
    set_linear_parameters!(model, θ)
    sys    = bulk(:Al, a=a0*u"Å", cubic=true) * supercell
    sys_md = Molly.System(sys; force_units=u"eV/Å", energy_units=u"eV")
    sys_md = Molly.System(sys_md;
        general_inters = (model,),
        velocities = Molly.random_velocities(sys_md, T),
        loggers = (temp   = Molly.TemperatureLogger(log_every),
                   coords = Molly.CoordinatesLogger(log_every),
                   volume = Molly.VolumeLogger(log_every),
                   energy = Molly.PotentialEnergyLogger(typeof(1.0u"eV"), log_every)))
    sim = Molly.Langevin(dt=dt, temperature=T, friction=friction,
                         coupling=Molly.MonteCarloBarostat(pressure, T, sys_md.boundary))
    Molly.simulate!(sys_md, sim, n_equil + n_prod)
    return sys_md
end

"Write every logged frame of an NPT run (cubic box, positions, energy, temperature, step) as extended XYZ."
function write_trajectory(file, sys_md; log_every=50)
    coords_hist = sys_md.loggers.coords.history
    vol_hist    = ustrip.(u"Å^3", sys_md.loggers.volume.history)
    temps_hist  = ustrip.(sys_md.loggers.temp.history)
    ener_hist   = ustrip.(sys_md.loggers.energy.history)
    n_atoms = length(coords_hist[1])
    side(f) = cbrt(vol_hist[f])
    species = [string(Molly.atomic_symbol(sys_md, i)) for i in 1:n_atoms]
    frames  = Dict{String,Any}[]
    for (f, fc) in enumerate(coords_hist)
        L = side(f)
        push!(frames, Dict{String,Any}(
            "N_atoms" => n_atoms,
            "info"    => Dict{String,Any}(
                "Lattice"     => "$L 0.0 0.0 0.0 $L 0.0 0.0 0.0 $L",
                "Properties"  => "species:S:1:pos:R:3",
                "energy"      => ener_hist[f], "temperature" => temps_hist[f],
                "step"        => (f-1)*log_every),
            "arrays"  => Dict{String,Any}(
                "species" => species,
                "pos"     => reduce(hcat, [ustrip.(u"Å", c) for c in fc]))))
    end
    ExtXYZ.write_frames(file, frames)
end

"Mean and standard deviation of the lattice constant ∛V / N_super over the production frames of an NPT run."
function lattice_constant_from_volume(sys_md; n_equil=10_000, log_every=50, N_super=4)
    vol_hist = ustrip.(u"Å^3", sys_md.loggers.volume.history)
    prod     = (div(n_equil, log_every)+1):length(vol_hist)
    a_prod   = cbrt.(vol_hist[prod]) ./ N_super
    return mean(a_prod), std(a_prod)
end

"""
    rdf_from_trajectory(file; n_equil=10_000, n_bins=200) -> (r, g)

The radial distribution function over the production frames (step ≥ n_equil) of a trajectory written by
`write_trajectory`: `n_bins` bins up to half the smallest box, each frame normalised with its own box.
"""
function rdf_from_trajectory(file; n_equil=10_000, n_bins=200)
    frames = ExtXYZ.read_frames(file)
    steps  = [Int(f["info"]["step"]) for f in frames]
    prod   = findall(>=(n_equil), steps)
    boxes = [frames[f]["cell"][1,1] for f in prod]
    poss  = [Matrix{Float64}(frames[f]["arrays"]["pos"]) for f in prod]
    r_max = minimum(boxes)/2
    dr    = r_max/n_bins
    r_mid = collect(range(dr/2, r_max-dr/2; length=n_bins))
    n_at = size(poss[1], 2)
    pair_counts = zeros(n_bins); ρacc = 0.0
    for (f, L) in enumerate(boxes)
        p = poss[f]; ρacc += n_at/L^3
        @inbounds for i in 1:n_at-1, j in i+1:n_at
            d1 = p[1,i]-p[1,j]; d2 = p[2,i]-p[2,j]; d3 = p[3,i]-p[3,j]
            d1 -= L*round(d1/L); d2 -= L*round(d2/L); d3 -= L*round(d3/L)
            r = sqrt(d1*d1 + d2*d2 + d3*d3)
            r < r_max || continue
            b = floor(Int, r/dr) + 1
            b <= n_bins && (pair_counts[b] += 2)
        end
    end
    ρbar = ρacc/length(boxes)
    g = [pair_counts[k]/(length(boxes)*n_at*4π*r_mid[k]^2*dr*ρbar) for k in 1:n_bins]
    return r_mid, g
end
