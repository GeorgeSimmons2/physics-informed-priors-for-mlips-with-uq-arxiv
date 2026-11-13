"""
Figure 3, stage 1: the MACE readout as a linear regression problem.

  constraint rows   derivatives of the readout basis of FCC Al at a0 = 4.032 Å with respect to the lattice
                    constant and to strain: dotted with θ they give the pressure and C11, C12, C44
  training rows     for every frame of the training set (data/Al/subset_20_percent.extxyz, the data the model was
                    trained on): the per-atom energy row and its target, and one row per force component

In:   models/mace/Al_small_stagetwo.model, data/Al/subset_20_percent.extxyz
Out (results/fig3/):
      theta0.csv, dD_da.csv, d2D_da2.csv, d2D_de11.csv, d2D_de12.csv, d2D_de44.csv, frozen.csv, meta.txt
      XE_train.csv, yE_train.csv              energy rows  D/N  and targets  (E_DFT − E_frozen)/N
      XF_train.csv, yF_train.csv              force rows  G = −∂D/∂r  and targets  F_DFT − F_frozen
      GtG_train.csv, Gty_train.csv, fstats_train.csv     the force rows' normal equations, summed frame by frame
Run:  THREADS=16 python scripts/fig3_mace_elastic/1_readout_rows.py             (about 20 min on 16 cores)
"""
import os
import sys
import time

import numpy as np
from ase.io import read

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import mace_readout as mr

mr.use_threads()
calc = mr.load_calc()
basis = mr.Basis(calc)
theta0 = basis.theta0
save = lambda name, arr: np.savetxt(mr.results("fig3", f"{name}.csv"), arr, delimiter=",")

# ── constraint rows at A0 ──────────────────────────────────────────────────
a_model, _ = mr.relaxed_a0(calc)
c0 = mr.fcc(mr.A0)
rows, frozen = mr.constraint_rows(basis, calc)
for name, arr in zip(["theta0", "dD_da", "d2D_da2", "d2D_de11", "d2D_de12", "d2D_de44", "frozen"], [theta0, *rows, frozen]):
    save(name, np.asarray(arr)[None, :])
with open(mr.results("fig3", "meta.txt"), "w") as f:
    f.write(f"a0 {mr.A0:.10f}\nV {c0.get_volume():.10f}\na_model {a_model:.10f}\nn_atoms {len(c0)}\nH_A 0.001\nH_E 0.0001\n")
k = mr.EV_A3_TO_GPA / c0.get_volume()
tot = frozen + np.array([r @ theta0 for r in rows])
print(f"MACE as trained: relaxed a0 = {a_model:.5f} Å; at A0 = {mr.A0} Å: C11 {k * tot[2]:.1f}, C12 {k * tot[3]:.1f}, "
      f"C44 {k * tot[4]:.1f} GPa, P {mr.EV_A3_TO_GPA * tot[0] / (3 * mr.A0**2):+.3f} GPa", flush=True)

# ── training rows ──────────────────────────────────────────────────────────
frames = read(os.path.join(mr.ROOT, "data", "Al", "subset_20_percent.extxyz"), ":")
t = time.time()
progress = lambda i: (i % 500 == 0 or i == len(frames) - 1) and print(f"  frame {i + 1}/{len(frames)}  {time.time() - t:.0f} s", flush=True)
train = mr.training_rows(basis, frames, progress)
for name, arr in train.items():
    save(f"{name}_train", arr)
print(f"{len(train['yE'])} energy rows and {len(train['yF'])} force rows → {mr.results('fig3', '')}")
