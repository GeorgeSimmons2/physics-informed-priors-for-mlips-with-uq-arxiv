#!/usr/bin/env python
"""
Vacancy-migration NEB barrier with two MACE models, "RLS" and "Constrained".

Labels: "RLS" is MACE as trained (the paper's name for the unconstrained fit; 6_neb_pops.py calls the same model
"MACE"), "Constrained" a checkpoint with an elastic-constrained readout written in (4_compare.py).

A standalone first look, superseded by 6_neb_pops.py, which runs the same NEB for both mean models AND every POPS
committee member in a cell fixed at the paper's A0.  The committed results in mace/results/neb_vacancy/ were run from
mace/ with --a 4.05 --relax-cell --fmax 0.03 against models/Al_small_stagetwo_elastic.model — the same C_ij constraints
imposed at A0 = 4.04 Å (physics-informed-priors-repro/mace_al_elastic), not the published θc, which is constrained at
4.032 Å (published/mace/results/Al_small_stagetwo_elastic.model; see README) — and --relax-cell relaxes the perfect
cell only to the same fmax as the atoms, so the two models ran at 4.05 Å ("RLS", no step taken) and 4.0425 Å
("Constrained") rather than at their own 4.052 / 4.040 Å.

For each model independently:
  1. build a bulk supercell, remove one atom (vacancy),
  2. make the final state by moving a nearest neighbour into the vacancy,
  3. relax both endpoints (fixed cell),
  4. run a climbing-image NEB (IDPP initial path),
  5. report forward/backward barriers and save the band.

Both profiles are then plotted on one figure against the cumulative path
length, so the two models can be compared on the same reaction coordinate.

Usage:
  cd mace && python vacancy_migration_barrier.py \
      --rls         path/to/rls.model \
      --constrained path/to/constrained.model \
      --element Al --lattice fcc --a 4.05 --size 3 \
      [--nimages 7 --fmax 0.03 --out results/neb_vacancy]

Defaults are for fcc Al; set --element/--lattice/--a for your system (the
lattice constant should be the one each model relaxes to - use --relax-cell
to let each model relax the perfect cell first).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ase.build import bulk
from ase.io import write
try:                                   # ASE >= 3.23
    from ase.mep import NEB, NEBTools
    from ase.filters import UnitCellFilter
except ImportError:                    # older ASE
    from ase.neb import NEB, NEBTools
    from ase.constraints import UnitCellFilter
from ase.optimize import BFGS, FIRE


def make_endpoints(element, lattice, a, size):
    """Perfect supercell, vacancy initial state, and the migrating-atom index.

    The vacancy is created at the atom nearest the cell centre. The migrating
    atom is one of its nearest neighbours; the final state is the initial
    state with that atom placed on the vacant site.
    """
    perfect = bulk(element, lattice, a=a, cubic=True).repeat((size, size, size))
    centre = perfect.cell.sum(axis=0) / 2
    v = int(np.argmin(np.linalg.norm(perfect.positions - centre, axis=1)))
    vac_pos = perfect.positions[v].copy()

    # nearest neighbour of the vacancy, by minimum-image distance
    d = perfect.get_distances(v, range(len(perfect)), mic=True)
    d[v] = np.inf
    j = int(np.argmin(d))

    initial = perfect.copy()
    del initial[v]
    # index of the migrating atom after deletion
    j_init = j if j < v else j - 1

    final = initial.copy()
    # move along the minimum-image vector so the hop never crosses the box
    vec = perfect.get_distance(j, v, mic=True, vector=True)
    final.positions[j_init] += vec
    # the migrating atom must now sit on the vacant site (modulo PBC)
    from ase.geometry import find_mic
    disp, _ = find_mic(final.positions[j_init][None] - vac_pos[None], final.cell, pbc=True)
    assert np.linalg.norm(disp) < 1e-8
    return perfect, initial, final, j_init, vac_pos


def relax(atoms, calc, fmax, logfile, relax_cell=False):
    atoms = atoms.copy()
    atoms.calc = calc
    target = UnitCellFilter(atoms) if relax_cell else atoms
    BFGS(target, logfile=logfile).run(fmax=fmax, steps=500)
    return atoms


def run_neb(label, model_path, args, outdir):
    from mace.calculators import MACECalculator

    calc = MACECalculator(model_paths=[model_path], device=args.device,
                          default_dtype="float64")
    log = outdir / f"{label}_opt.log"

    perfect, initial, final, mig, _ = make_endpoints(
        args.element, args.lattice, args.a, args.size)

    if args.relax_cell:
        # let this model choose its own lattice constant, then rebuild
        p = relax(perfect, calc, args.fmax, log, relax_cell=True)
        a_model = p.cell.lengths()[0] / args.size
        print(f"[{label}] relaxed lattice constant: {a_model:.4f} A")
        perfect, initial, final, mig, _ = make_endpoints(
            args.element, args.lattice, a_model, args.size)
    else:
        a_model = args.a

    initial = relax(initial, calc, args.fmax, log)
    final = relax(final, calc, args.fmax, log)
    e_perfect = perfect.copy(); e_perfect.calc = calc
    e_form = (initial.get_potential_energy()
              - (len(initial) / len(perfect)) * e_perfect.get_potential_energy())
    print(f"[{label}] vacancy formation energy: {e_form:.4f} eV")

    images = [initial] + [initial.copy() for _ in range(args.nimages - 2)] + [final]
    for im in images[1:-1]:
        im.calc = calc
    neb = NEB(images, climb=True, k=args.k, allow_shared_calculator=True)
    neb.interpolate(method="idpp")

    opt = FIRE(neb, logfile=str(outdir / f"{label}_neb.log"))
    opt.run(fmax=args.fmax, steps=args.steps)

    energies = np.array([im.get_potential_energy() for im in images])
    tools = NEBTools(images)
    fit = tools.get_fit()
    barrier_fwd, dE = tools.get_barrier(fit=False)
    barrier_bwd = barrier_fwd - dE
    # NEBTools.get_fmax rebuilds a NEB without allow_shared_calculator, so
    # take the residual band force from our own NEB object instead.
    fmax_band = float(np.sqrt((neb.get_forces() ** 2).sum(axis=1).max()))
    print(f"[{label}] barrier forward {barrier_fwd:.4f} eV, "
          f"backward {barrier_bwd:.4f} eV, dE {dE:.4f} eV, "
          f"max |F| on band {fmax_band:.4f} eV/A")

    write(outdir / f"{label}_band.xyz", images)
    return {
        "label": label,
        "model": str(model_path),
        "lattice_constant": float(a_model),
        "n_atoms": len(initial),
        "migrating_atom": int(mig),
        "vacancy_formation_eV": float(e_form),
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
           title="Vacancy migration, climbing-image NEB")
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
    ap.add_argument("--lattice", default="fcc", choices=["fcc", "bcc", "sc"])
    ap.add_argument("--a", type=float, default=4.05, help="lattice constant / A")
    ap.add_argument("--size", type=int, default=3, help="supercell repeats of the cubic cell")
    ap.add_argument("--relax-cell", action="store_true",
                    help="relax the perfect cell with each model before building the vacancy")
    ap.add_argument("--nimages", type=int, default=7, help="total images incl. endpoints")
    ap.add_argument("--k", type=float, default=0.1, help="NEB spring constant / eV A^-2")
    ap.add_argument("--fmax", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/neb_vacancy")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    results = [run_neb("RLS", args.rls, args, outdir),
               run_neb("Constrained", args.constrained, args, outdir)]

    with open(outdir / "neb_results.json", "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    plot(results, outdir / "neb_vacancy.png")

    r, c = results
    print(f"\nbarrier  RLS: {r['barrier_forward_eV']:.4f} eV   "
          f"Constrained: {c['barrier_forward_eV']:.4f} eV   "
          f"difference: {(c['barrier_forward_eV'] - r['barrier_forward_eV'])*1e3:+.1f} meV")


if __name__ == "__main__":
    main()
