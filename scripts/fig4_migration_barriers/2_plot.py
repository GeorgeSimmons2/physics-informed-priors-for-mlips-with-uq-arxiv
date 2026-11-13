"""
Figure 4: migration barriers of MACE (orange) and of the constrained MACE (blue) with POPS uncertainty.
(a) vacancy, (b) <100> split-dumbbell interstitial.  Line: the mean model's band.  Dark band: ± one standard
deviation of the committee; light band: its full range.  The legend gives the mean model's barrier ± the standard
deviation of the members' barriers.

In:   results/fig4/bands/ from stage 1 (collected into results/fig4/neb_pops.npz), else published/fig4/neb_pops.npz
Out:  results/fig4/neb_pops.{npz,pdf,png}
Run:  python scripts/fig4_migration_barriers/2_plot.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import mace_readout as mr

DEFECTS = ("vacancy", "interstitial")
MODELS = ("unconstrained", "constrained")
LABEL = {"unconstrained": "MACE", "constrained": "constrained"}
COL = {"unconstrained": "C1", "constrained": "C0"}

# ── collect this run's bands, if there are any ─────────────────────────────
bands = mr.results("fig4", "bands", "")
if os.path.exists(os.path.join(bands, "vacancy_unconstrained_mean.npz")):
    res = {}
    for defect in DEFECTS:
        for name in MODELS:
            stem = os.path.join(bands, f"{defect}_{name}")
            mean = np.load(stem + "_mean.npz")
            members = [np.load(f"{stem}_m{i}.npz") for i in range(1000) if os.path.exists(f"{stem}_m{i}.npz")]
            frac = mean["frac"]
            res.update({f"{defect}_{name}_frac": frac, f"{defect}_{name}_rel_mean": mean["rel"],
                        f"{defect}_{name}_barrier": float(mean["barrier"]), f"{defect}_{name}_a0": float(mean["a0"]),
                        f"{defect}_{name}_a0_relaxed": np.array([float(z["a0"]) for z in members]),
                        f"{defect}_{name}_barrier_relaxed": np.array([float(z["barrier"]) for z in members]),
                        f"{defect}_{name}_rel_relaxed": np.array([np.interp(frac, z["frac"], z["rel"]) for z in members])})
    np.savez(mr.results("fig4", "neb_pops.npz"), **res)
d = np.load(mr.input_file("fig4", "neb_pops.npz"))

# ── figure ─────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10})
fig, axes = plt.subplots(1, len(DEFECTS), figsize=(3.54 * len(DEFECTS), 3.4), squeeze=False)
for ax, defect, tag in zip(axes[0], DEFECTS, "abcd"):
    for name in MODELS:
        get = lambda key: d[f"{defect}_{name}_{key}"]
        x, M = get("frac"), 1e3 * get("rel_relaxed")
        if len(M):
            ax.fill_between(x, M.min(0), M.max(0), color=COL[name], alpha=0.12, lw=0)
            ax.fill_between(x, M.mean(0) - M.std(0), M.mean(0) + M.std(0), color=COL[name], alpha=0.3, lw=0)
        b = 1e3 * get("barrier_relaxed")
        ax.plot(x, 1e3 * get("rel_mean"), "-o", color=COL[name], ms=3.5, lw=1.5,
                label=f"{LABEL[name]}: {1e3 * float(get('barrier')):.0f} ± {b.std() if len(b) else 0:.0f} meV")
        print(f"{defect:12s} {LABEL[name]:11s} mean model {1e3 * float(get('barrier')):6.1f} meV, members {b.mean():6.1f} ± {b.std():.1f} meV (n = {len(b)})")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("fractional path length"); ax.set_ylabel(r"$E - E_\mathrm{initial}$ (meV)")
    ax.set_title(f"({tag}) {defect} migration", loc="left")
    ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(mr.results("fig4", f"neb_pops.{ext}"), dpi=200)
print(f"figure → {mr.results('fig4', 'neb_pops')}.pdf/.png")
