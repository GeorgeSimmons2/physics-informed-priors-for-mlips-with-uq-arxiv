"""
Figure 3, stage 1a: the constraint rows of the MACE readout.

Derivatives of the readout basis of FCC Al at a0 = 4.032 Å with respect to the lattice constant and to strain:
dotted with the readout weights θ they give the pressure and C11, C12, C44.

In:   models/mace/Al_small_stagetwo.model
Out:  results/fig3/theta0.csv, dD_da.csv, d2D_da2.csv, d2D_de11.csv, d2D_de12.csv, d2D_de44.csv, frozen.csv, meta.txt
Run:  THREADS=16 python scripts/fig3_mace_elastic/1a_constraint_rows.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import mace_readout as mr

mr.use_threads()
calc = mr.load_calc()
basis = mr.Basis(calc)
theta0 = basis.theta0
save = lambda name, arr: np.savetxt(mr.results("fig3", f"{name}.csv"), arr, delimiter=",")

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
