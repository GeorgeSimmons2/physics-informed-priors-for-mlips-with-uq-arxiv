"""
The library behind the figures of "Physics Informed Priors for Machine Learning Interatomic Potentials with
Uncertainty Quantification" (arXiv:2611.09325).

Everything a constraint needs is linear in the model parameters θ: the energy of a strained cell, the dynamical
matrix at a wave vector, the pressure.  Each file builds such rows once and the rest is linear algebra on θ.

    paths.jl            where data, models, results and the published results live
    models.jl           load and fit the two ACE models; the regression problem behind a fit
    elasticity.jl       lattice constant, lattice and elastic rows, Born stability
    phonons.jl          force constants, the per-basis ("undotted") Hessian, band paths, bands, phonon stability
    pops.jl             POPS: pointwise corrections, the hypercube, sampling with and without rejection
    constrained_fit.jl  the constrained least-squares fit and its phonon cutting planes
    md.jl               the NPT run and its radial distribution function
    readout.jl          the MACE readout: constrained refit and POPS clouds (the Julia half of Figures 3 and 4)
"""
module PhysicsPriors

using LinearAlgebra, Statistics, Random, Printf, DelimitedFiles, Serialization, SparseArrays
using StaticArrays, Unitful, ForwardDiff, OSQP
using ACEpotentials, ACEfit, ExtXYZ
using AtomsBase, AtomsBuilder, GeometryOptimization
using AtomsBuilder: bulk
using Bumper: @no_escape
import ACEpotentials.Models: evaluate_basis, set_linear_parameters!
using AtomsCalculatorsUtilities.SitePotentials: hessian
import AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius
import Molly

include("paths.jl")
include("models.jl")
include("elasticity.jl")
include("phonons.jl")
include("pops.jl")
include("constrained_fit.jl")
include("md.jl")
include("readout.jl")

export root, results, published, input
export AL12, AL16, model_dir, load_model, fit_model, regression_problem
export relax_lattice_constant, relax_or_fallback, lattice_rows, cubic_elastic_rows, elastic_constants, lattice_shift
export born_constraints, is_born_stable, lattice_drift_ok, positive_curvature
export FREQ_THz, bulk_prim_super, precompute_force_constants, dynamical_matrix_from_fc, fcc_band_path
export undotted_hessian, undotted_hessian_rows, bandpath_basis, bandpath_basis_rows, bands, bands_record, native_bands
export map_with_model_copies
export omega2_cut, min_freq_stable, soft_modes, cut_row, is_phonon_stable
export corrections, hypercube, sample_hypercube, rejection_sample_hypercube
export top_leverage_rows, pinned_corrections
export qp_matrices, constrained_fit!, fit_with_phonon_cuts, OSQP_AL12, OSQP_RLS
export run_npt, write_trajectory, lattice_constant_from_volume, rdf_from_trajectory
export readout_inputs, readout_fit, readout_clouds

end
