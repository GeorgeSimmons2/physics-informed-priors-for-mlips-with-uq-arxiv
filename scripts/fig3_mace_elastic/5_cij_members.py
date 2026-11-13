"""
Figure 3, stage 5: the lattice constant and elastic constants of every committee member.

Each member's θ is written into the MACE readout; its lattice constant is relaxed and C11, C12, C44 follow from
the analytic stress under ±0.5 % strain at that lattice constant.  The same for the two reference models: the
constrained readout θc and MACE as trained, θ0.

In:   results/fig3/committee_{constrained,unconstrained}.csv, theta_constrained.csv, theta0.csv
Out:  results/fig3/cij_members.csv      one row per model: committee, member (−1 = reference), a0, C11, C12, C44, B
Run:  THREADS=16 python scripts/fig3_mace_elastic/5_cij_members.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import mace_readout as mr

mr.use_threads()
load = lambda *parts: np.loadtxt(mr.input_file(*parts), delimiter=",")
NAMES = ("constrained", "unconstrained")
committee = {n: load("fig3", f"committee_{n}.csv") for n in NAMES}
reference = {"constrained": load("fig3", "theta_constrained.csv").ravel(), "unconstrained": load("fig3", "theta0.csv").ravel()}

calc = mr.load_calc()
rows = []
for j, name in enumerate(NAMES):
    for i, theta in [(-1, reference[name])] + list(enumerate(committee[name])):
        mr.set_theta(calc, theta)
        r = mr.stress_strain_cij(calc)
        rows.append([j, i, r["a0"], r["C11"], r["C12"], r["C44"], r["B"]])
rows = np.array(rows)
np.savetxt(mr.results("fig3", "cij_members.csv"), rows, delimiter=",",
           header="committee (0 constrained, 1 unconstrained),member (-1 = reference model),a0 (A),C11,C12,C44,B (GPa)")

for j, name in enumerate(NAMES):
    r, r0 = rows[(rows[:, 0] == j) & (rows[:, 1] >= 0)], rows[(rows[:, 0] == j) & (rows[:, 1] < 0)][0]
    print(f"{name:14s} reference a0 {r0[2]:.4f} C11 {r0[3]:.1f} C12 {r0[4]:.1f} C44 {r0[5]:.1f} | members a0 "
          f"{r[:, 2].min():.4f}–{r[:, 2].max():.4f}  C11 {r[:, 3].min():.1f}–{r[:, 3].max():.1f}  "
          f"C12 {r[:, 4].min():.1f}–{r[:, 4].max():.1f}  C44 {r[:, 5].min():.1f}–{r[:, 5].max():.1f} GPa")
