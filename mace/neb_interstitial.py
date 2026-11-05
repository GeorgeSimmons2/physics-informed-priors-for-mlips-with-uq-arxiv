#!/usr/bin/env python
"""
Self-interstitial migration NEB barrier in fcc Al with two MACE models,
"RLS" and "Constrained".

Labels: "RLS" is MACE as trained (the paper's name for the unconstrained fit; 6_neb_pops.py calls the same model
"MACE"), "Constrained" a checkpoint with an elastic-constrained readout written in (4_compare.py).

A standalone first look, superseded by 6_neb_pops.py (both mean models and every POPS committee member, cell fixed
at the paper's A0).  The committed results in mace/results/neb_interstitial/ were run from mace/ with --a 4.05
--relax-cell --fmax 0.03 against models/Al_small_stagetwo_elastic.model, the same C_ij constraints imposed at
A0 = 4.04 Å rather than the published θc at 4.032 Å (see README and vacancy_migration_barrier.py).

The stable self-interstitial in fcc Al is the <100> split dumbbell, and its
lowest-energy migration is the translation-rotation jump: one atom of a [100]
dumbbell centred on site A hops to the nearest-neighbour site B = A + a/2[110],
pushing the atom at B aside so that the two form a new dumbbell at B whose axis
has rotated by 90 degrees to [010]. Atom bookkeeping between the endpoints:

    initial                          final
    a1 = A + d x        ->           B - d y     (arriving atom)
    lattice atom at B   ->           B + d y     (pushed aside)
    a2 = A - d x        ->           A           (left behind, back on site)

so that the IDPP interpolation follows the physical mechanism.

For each model independently: build the supercell, construct and relax both
dumbbell endpoints (fixed cell, optionally after relaxing the perfect cell),
run a climbing-image NEB, report formation energy and barrier, save the band.
Both profiles are plotted on one figure.

Usage:
  cd mace && python neb_interstitial.py \
      --rls         models/Al_small_stagetwo.model \
      --constrained models/Al_small_stagetwo_elastic.model \
      --a 4.05 --size 3 --relax-cell [--nimages 7 --fmax 0.03 --out results/neb_interstitial]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ase import Atoms
from ase.build import bulk
from ase.io import write
try:                                   # ASE >= 3.23
    from ase.mep import NEB, NEBTools
    from ase.filters import UnitCellFilter
except ImportError:                    # older ASE
    from ase.neb import NEB, NEBTools
    from ase.constraints import UnitCellFilter
from ase.optimize import BFGS, FIRE


def make_endpoints(element, a, size, delta):
    """Perfect supercell plus the initial/final <100> dumbbell configurations.

    delta is the half-separation of the dumbbell (Angstrom); the relaxation
    fixes the exact value, it only needs to be close enough to fall into the
    dumbbell basin.
    """
    perfect = bulk(element, "fcc", a=a, cubic=True).repeat((size, size, size))
    pos = perfect.positions
    centre = perfect.cell.sum(axis=0) / 2
    A = int(np.argmin(np.linalg.norm(pos - centre, axis=1)))

    # nearest neighbour of A along +[110]: minimum-image vector closest to a/2 (1,1,0)
    vecs = perfect.get_distances(A, range(len(perfect)), mic=True, vector=True)
    target = 0.5 * a * np.array([1.0, 1.0, 0.0])
    dev = np.linalg.norm(vecs - target, axis=1)
    dev[A] = np.inf
    B = int(np.argmin(dev))
    assert dev[B] < 1e-6, "could not find the [110] neighbour; is the cell cubic fcc?"
    rA = pos[A].copy()
    rB = rA + target                     # unwrapped position of site B
    x, y = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])

    others = [i for i in range(len(perfect)) if i not in (A, B)]
    init_pos = np.vstack([pos[others], [rB], [rA + delta * x], [rA - delta * x]])
    final_pos = np.vstack([pos[others], [rB + delta * y], [rB - delta * y], [rA]])
    symbols = [element] * len(init_pos)
    initial = Atoms(symbols, positions=init_pos, cell=perfect.cell, pbc=True)
    final = Atoms(symbols, positions=final_pos, cell=perfect.cell, pbc=True)
    return perfect, initial, final, {"A": A, "B": B, "site_A": rA.tolist(),
                                     "site_B": rB.tolist()}


def relax(atoms, calc, fmax, logfile, relax_cell=False):
    atoms = atoms.copy()
    atoms.calc = calc
    target = UnitCellFilter(atoms) if relax_cell else atoms
    BFGS(target, logfile=logfile).run(fmax=fmax, steps=1000)
    return atoms


def dumbbell_geometry(atoms, idx_pair):
    """Separation and unit axis of the two atoms forming the dumbbell."""
    v = atoms.get_distance(*idx_pair, mic=True, vector=True)
    d = np.linalg.norm(v)
    return float(d), (v / d).tolist()


def run_neb(label, model_path, args, outdir):
    from mace.calculators import MACECalculator

    calc = MACECalculator(model_paths=[model_path], device=args.device,
                          default_dtype="float64")
    log = str(outdir / f"{label}_opt.log")

    perfect, initial, final, info = make_endpoints(args.element, args.a, args.size, args.delta)
    a_model = args.a
    if args.relax_cell:
        p = relax(perfect, calc, args.fmax, log, relax_cell=True)
        a_model = float(p.cell.lengths()[0] / args.size)
        print(f"[{label}] relaxed lattice constant: {a_model:.4f} A")
        perfect, initial, final, info = make_endpoints(args.element, a_model, args.size, args.delta)

    n = len(initial)
    initial = relax(initial, calc, args.fmax, log)
    final = relax(final, calc, args.fmax, log)
    e_perfect = perfect.copy(); e_perfect.calc = calc
    e_form = (initial.get_potential_energy()
              - (n / len(perfect)) * e_perfect.get_potential_energy())
    d_init, ax_init = dumbbell_geometry(initial, (n - 2, n - 1))   # a1, a2
    d_fin, ax_fin = dumbbell_geometry(final, (n - 3, n - 2))       # B+dy, B-dy
    print(f"[{label}] interstitial formation energy: {e_form:.4f} eV; "
          f"dumbbell separation {d_init:.3f} A (initial), {d_fin:.3f} A (final)")
    if abs(abs(ax_init[0]) - 1) > 0.1 or abs(abs(ax_fin[1]) - 1) > 0.1:
        print(f"[{label}] WARNING: relaxed dumbbell axes {np.round(ax_init, 2)} / "
              f"{np.round(ax_fin, 2)} are not [100]/[010]; the endpoints may have "
              f"relaxed into a different interstitial configuration.")

    images = [initial] + [initial.copy() for _ in range(args.nimages - 2)] + [final]
    for im in images[1:-1]:
        im.calc = calc
    neb = NEB(images, climb=True, k=args.k, allow_shared_calculator=True)
    neb.interpolate(method="idpp")

    opt = FIRE(neb, logfile=str(outdir / f"{label}_neb.log"))
    opt.run(fmax=args.fmax, steps=args.steps)

    energies = np.array([im.get_potential_energy() for im in images])
    tools = NEBTools(images)
    try:
        from ase.utils.forcecurve import fit_images
        fit = fit_images(images)
    except ImportError:
        fit = tools.get_fit()
    barrier_fwd, dE = tools.get_barrier(fit=False)
    barrier_bwd = barrier_fwd - dE
    # NEBTools.get_fmax rebuilds a NEB without allow_shared_calculator, so
    # take the residual band force from our own NEB object instead.
    fmax_band = float(np.sqrt((neb.get_forces() ** 2).sum(axis=1).max()))
    print(f"[{label}] barrier forward {barrier_fwd:.4f} eV, backward {barrier_bwd:.4f} eV, "
          f"dE {dE:.4f} eV (should be ~0 by symmetry), max |F| on band {fmax_band:.4f} eV/A")

    write(outdir / f"{label}_band.xyz", images)
    return {
        "label": label,
        "model": str(model_path),
        "mechanism": "<100> dumbbell translation-rotation jump to [110] neighbour",
        "lattice_constant": a_model,
        "n_atoms": n,
        "sites": info,
        "dumbbell_separation_initial_A": d_init,
        "dumbbell_axis_initial": ax_init,
        "dumbbell_separation_final_A": d_fin,
        "dumbbell_axis_final": ax_fin,
        "formation_energy_eV": float(e_form),
        "barrier_forward_eV": float(barrier_fwd),
        "barrier_backward_eV": float(barrier_bwd),
        "dE_eV": float(dE),
        "neb_converged": bool(opt.nsteps < args.steps),
        "fmax_band_eV_A": fmax_band,
        "path": list(map(float, fit.path)),
        "energies_rel": (energies - energies[0]).tolist(),
        "fit_path": list(map(float, fit.fit_path)),
        "fit_energies": list(map(float, fit.fit_energies)),
    }


def plot(results, out):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = {"RLS": "#1f78ff", "Constrained": "#ff6347"}
    for r in results:
        c = colors.get(r["label"], None)
        ax.plot(r["fit_path"], r["fit_energies"], "-", color=c, lw=1.5,
                label=f"{r['label']}: {r['barrier_forward_eV']*1e3:.0f} meV")
        ax.plot(r["path"], r["energies_rel"], "o", color=c, ms=5)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set(xlabel="Path length / A", ylabel="Energy relative to initial / eV",
           title="<100> dumbbell migration in fcc Al, climbing-image NEB")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rls", required=True, help="RLS .model path")
    ap.add_argument("--constrained", required=True, help="Constrained .model path")
    ap.add_argument("--element", default="Al")
    ap.add_argument("--a", type=float, default=4.05, help="fcc lattice constant / A")
    ap.add_argument("--size", type=int, default=3, help="supercell repeats of the cubic cell")
    ap.add_argument("--delta", type=float, default=0.85,
                    help="initial dumbbell half-separation / A (relaxation refines it)")
    ap.add_argument("--relax-cell", action="store_true",
                    help="relax the perfect cell with each model before building the defect")
    ap.add_argument("--nimages", type=int, default=7, help="total images incl. endpoints")
    ap.add_argument("--k", type=float, default=0.1, help="NEB spring constant / eV A^-2")
    ap.add_argument("--fmax", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/neb_interstitial")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    results = [run_neb("RLS", args.rls, args, outdir),
               run_neb("Constrained", args.constrained, args, outdir)]

    with open(outdir / "neb_results.json", "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    plot(results, outdir / "neb_interstitial.png")

    r, c = results
    print(f"\nbarrier  RLS: {r['barrier_forward_eV']:.4f} eV   "
          f"Constrained: {c['barrier_forward_eV']:.4f} eV   "
          f"difference: {(c['barrier_forward_eV'] - r['barrier_forward_eV'])*1e3:+.1f} meV")
    print(f"formation  RLS: {r['formation_energy_eV']:.4f} eV   "
          f"Constrained: {c['formation_energy_eV']:.4f} eV")


if __name__ == "__main__":
    main()
