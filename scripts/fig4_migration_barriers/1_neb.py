"""
Figure 4, stage 1: vacancy and interstitial migration barriers of MACE and of the constrained MACE, each with its
POPS committee of 100 members.

For each defect and each model: the mean model's band first (cached), then every committee member's band from it.
Every model and every member is in its own relaxed cell (python/neb.py).  Members are independent, so the work
can be split across jobs with MEMBERS=a-b; each band is written once and never recomputed.

In:   models/mace/Al_small_stagetwo.model, results/fig3/theta_constrained.csv, committee_{constrained,unconstrained}.csv
Out:  results/fig4/bands/<defect>_<model>_mean.{xyz,npz}, <defect>_<model>_m<i>.npz
Run:  THREADS=16 python scripts/fig4_migration_barriers/1_neb.py                (about 10 h on 16 cores)
      MEMBERS=0-49 THREADS=16 python scripts/fig4_migration_barriers/1_neb.py   (one of several jobs)
"""
import os
import sys

import numpy as np
from ase.io import read, write

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import mace_readout as mr
import neb

N_MEMBERS = 100
DEFECTS = ("vacancy", "interstitial")
MODELS = ("unconstrained", "constrained")            # MACE as trained; the elastic-constrained readout
first, last = (int(x) for x in os.environ.get("MEMBERS", f"0-{N_MEMBERS - 1}").split("-"))

mr.use_threads()
load = lambda *parts: np.loadtxt(mr.input_file(*parts), delimiter=",")
theta = {"unconstrained": mr.Basis(mr.load_calc()).theta0, "constrained": load("fig3", "theta_constrained.csv").ravel()}
committee = {name: load("fig3", f"committee_{name}.csv")[:N_MEMBERS] for name in MODELS}
calcs = {name: mr.load_calc() for name in MODELS}
log = mr.results("fig4", "bands", f"opt_{first}-{last}.log")
open(log, "w").close()


def summary(E, images):
    rel = E - E[0]
    return dict(E=E, rel=rel, frac=neb.path_frac(images), barrier=rel.max())


for defect in DEFECTS:
    for name in MODELS:
        calc = calcs[name]
        mr.set_theta(calc, theta[name])
        stem = mr.results("fig4", "bands", f"{defect}_{name}")
        if os.path.exists(stem + "_mean.npz"):
            images, a_mean = read(stem + "_mean.xyz", ":"), float(np.load(stem + "_mean.npz")["a0"])
        else:
            images, E, a_mean = neb.mean_band(defect, calc, log)
            write(stem + "_mean.xyz", images)
            np.savez(stem + "_mean.npz", a0=a_mean, **summary(E, images))
        print(f"{defect:12s} {name:13s} mean model: a0 {a_mean:.4f} Å, barrier {1e3 * float(np.load(stem + '_mean.npz')['barrier']):.1f} meV", flush=True)
        for i in range(first, min(last, len(committee[name]) - 1) + 1):
            if os.path.exists(f"{stem}_m{i}.npz"):
                continue
            images_i, E, a, converged = neb.member_band(committee[name][i], images, a_mean, calc, log)
            np.savez(f"{stem}_m{i}.npz", a0=a, converged=converged, **summary(E, images_i))
            print(f"    m{i:<3d} a0 {a:.4f} Å  barrier {1e3 * (E - E[0]).max():7.1f} meV{'' if converged else '  NOT CONVERGED'}", flush=True)
        mr.set_theta(calc, theta[name])
