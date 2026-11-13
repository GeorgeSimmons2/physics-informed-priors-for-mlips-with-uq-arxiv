"""
Figure 3: lattice constant and elastic constants of the constrained MACE readout and its POPS committee (blue)
against MACE as trained and its committee (orange).  Stars: the two reference models.  Lines: the targets
a0 = 4.032 Å, C11 = 116.3, C12 = 64.8, C44 = 30.9 GPa.

In:   results/fig3/cij_members.csv       (published/ if this run has none)
Out:  results/fig3/cij_spread.{pdf,png}
Run:  python scripts/fig3_mace_elastic/6_plot.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import mace_readout as mr

cij = np.loadtxt(mr.input_file("fig3", "cij_members.csv"), delimiter=",")
W = 255 / 72                                    # half the text width (510 pt), in inches
COL = {"constrained": "C0", "unconstrained": "C1"}
TOL = 1e-3                                      # the ±0.1 % constraint windows
FONT = 6

with plt.rc_context({"font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
                     "xtick.labelsize": FONT, "ytick.labelsize": FONT}):
    fig, axes = plt.subplots(2, 2, figsize=(W, W))
    rng = np.random.default_rng(0)              # horizontal jitter of the members
    for ax, (n, key, ref) in zip(axes.flat, ((2, "$a_0$ (Å)", 4.032), (3, "$C_{11}$ (GPa)", 116.3),
                                           (4, "$C_{12}$ (GPa)", 64.8), (5, "$C_{44}$ (GPa)", 30.9))):
        for j, name in enumerate(COL):
            m = (cij[:, 0] == j) & (cij[:, 1] >= 0)
            r0 = cij[(cij[:, 0] == j) & (cij[:, 1] < 0)][0]
            ax.plot(np.full(m.sum(), j) + rng.uniform(-0.18, 0.18, m.sum()), cij[m, n], ".", ms=2.5, alpha=0.6, color=COL[name])
            ax.plot([j], [r0[n]], marker="*", ms=11, color=COL[name], mec="k", mew=0.6, ls="none")
        if n > 2:
            ax.axhspan(ref * (1 - TOL), ref * (1 + TOL), color="k", alpha=0.25, lw=0)
        ax.axhline(ref, color="k", lw=0.7)
        ax.set_xticks([0, 1], ["constr.", "MACE"]); ax.set_xlim(-0.6, 1.6)
        ax.set_title(key)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(mr.results("fig3", f"cij_spread.{ext}"), dpi=300)
print(f"figure → {mr.results('fig3', 'cij_spread')}.pdf/.png")
