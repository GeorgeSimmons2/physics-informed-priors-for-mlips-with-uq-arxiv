"""
Figure 3, stage 1b: the training set as regression rows for the MACE readout.

For every frame of the training set (data/Al/subset_20_percent.extxyz, the data the model was trained on): the
per-atom energy row and its target, and one row per force component.

This is its own script, run in a fresh process, on purpose.  MACE's first evaluations in a process differ from
later ones in the last digit, and the POPS cloud built from these rows is sensitive to it: computed after the
constraint rows in the same process, the first frame's six force rows change by one unit in the last place and
the committees of stage 4 come out different.

In:   models/mace/Al_small_stagetwo.model, data/Al/subset_20_percent.extxyz
Out (results/fig3/):
      XE_train.csv, yE_train.csv              energy rows  D/N  and targets  (E_DFT − E_frozen)/N
      XF_train.csv, yF_train.csv              force rows  G = −∂D/∂r  and targets  F_DFT − F_frozen
      GtG_train.csv, Gty_train.csv, fstats_train.csv     the force rows' normal equations, summed frame by frame
Run:  THREADS=16 python scripts/fig3_mace_elastic/1b_training_rows.py             (about 20 min on 16 cores)
"""
import os
import sys
import time

import numpy as np
from ase.io import read

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import mace_readout as mr

mr.use_threads()
basis = mr.Basis(mr.load_calc())

frames = read(os.path.join(mr.ROOT, "data", "Al", "subset_20_percent.extxyz"), ":")
t = time.time()
progress = lambda i: (i % 500 == 0 or i == len(frames) - 1) and print(f"  frame {i + 1}/{len(frames)}  {time.time() - t:.0f} s", flush=True)
train = mr.training_rows(basis, frames, progress)
for name, arr in train.items():
    np.savetxt(mr.results("fig3", f"{name}_train.csv"), arr, delimiter=",")
print(f"{len(train['yE'])} energy rows and {len(train['yF'])} force rows → {mr.results('fig3', '')}")
