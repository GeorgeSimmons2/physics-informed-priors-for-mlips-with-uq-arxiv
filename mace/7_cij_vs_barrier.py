"""
7. Elastic constants against migration barrier, member by member: does the spread of a POPS committee's barriers
follow the spread of its elastic constants?

For each committee (constrained, about θc; unconstrained, about MACE as trained) and each member i:
  x  a0, C11, C12, C44 (and B) from $OUT/mace/pops/cij_members.csv       p4_evaluate.py: analytic stress–strain at the
                                                                           member's OWN relaxed a0
  y  the member's climbing-image NEB barrier from $OUT/mace/results/neb_pops.npz
                                                                           6_neb_pops.py --collect: the band re-converged
                                                                           by the member between its committee's mean-model
                                                                           endpoints (or its own: ENDPOINTS), cell at A0
Members are matched by index — row i of committee_<name>.csv in both files.  Each panel shows one committee's members,
its reference model (star), the paper's target (vertical line; ±0.1 % window for the C_ij), a least-squares line, and
Pearson r / Spearman ρ.  The table always covers both committees; the figure draws the committees in COMMITTEES
(default: unconstrained only — the constrained members' C_ij spread is ~0.01 GPa, the ±0.1 % windows, so their row
carries no information; COMMITTEES=unconstrained,constrained draws both, one row each with its own x-axes).

Read the unconstrained row with care: its barriers are computed in the A0 = 4.032 Å cell while each member's own a0 is
4.04–4.07 Å, so part of its barrier spread is residual pressure (volume), not elasticity — hence the a0 column.

Run: python3 7_cij_vs_barrier.py          Env: OUT (repro/saved), DEFECTS (vacancy | vacancy,interstitial),
     COMMITTEES (unconstrained | unconstrained,constrained: rows of the figure),
     REPLOT_FROM (read both inputs from another route's mace/ instead of $OUT/mace)
Out ($OUT/mace/results): cij_vs_barrier_<defect>.{pdf,png,txt}
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ("unconstrained", "constrained")                         # order in the table
ROWS = tuple(os.environ.get("COMMITTEES", "unconstrained").split(","))   # rows of the figure
assert all(r in NAMES for r in ROWS), f"COMMITTEES={ROWS}: from {NAMES}"
LABEL = {"constrained": "constrained", "unconstrained": "MACE"}
COL = {"constrained": "C0", "unconstrained": "C1"}
# name, axis label, the paper's target, column of cij_members.csv (committee, member, a0, C11, C12, C44, B)
QUANT = (("a0", "$a_0$ (Å)", 4.032, 2), ("C11", "$C_{11}$ (GPa)", 116.3, 3), ("C12", "$C_{12}$ (GPa)", 64.8, 4),
         ("C44", "$C_{44}$ (GPa)", 30.9, 5), ("B", "$B$ (GPa)", (116.3 + 2 * 64.8) / 3, 6))
PANELS = QUANT[:4]                                               # a0, C11, C12, C44 drawn; B only in the table
TOL = 1e-3                                                       # the constraint windows, as 3_qp.jl
W = 510 / 72                                                     # the MLST text width, in inches


def pearson(x, y):
    """Pearson correlation; nan if either variable is constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sx, sy = x.std(), y.std()
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)) if sx > 0 and sy > 0 else float("nan")


def spearman(x, y):
    """Spearman rank correlation (no tie handling: the inputs are continuous)."""
    rank = lambda v: np.argsort(np.argsort(np.asarray(v))).astype(float)
    return pearson(rank(x), rank(y))


def linfit(x, y):
    """Least-squares slope and intercept of y on x."""
    slope, intercept = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return float(slope), float(intercept)


def load_cij(path):
    """cij_members.csv → ({committee: (member indices, their rows)}, {committee: the reference model's row})."""
    rows = np.loadtxt(path, delimiter=",")
    out, ref = {}, {}
    for j, name in enumerate(("constrained", "unconstrained")):       # p4's codes: 0 constrained, 1 unconstrained
        r = rows[rows[:, 0] == j]
        m = r[:, 1] >= 0
        out[name] = (r[m, 1].astype(int), r[m])
        ref[name] = r[~m][0]
    return out, ref


def load_barriers(path, defect):
    """neb_pops.npz → ({committee: (member indices, relaxed barriers in meV)}, {committee: the mean model's barrier in meV})."""
    z = np.load(path)
    out, mean = {}, {}
    for name in NAMES:
        k = f"{defect}_{name}_"
        if k + "members" not in z.files:
            raise KeyError(f"{path} has no {defect}/{name} results (DEFECTS of 6_neb_pops.py?)")
        out[name] = (z[k + "members"].astype(int), 1e3 * np.asarray(z[k + "barrier_relaxed"], float))
        mean[name] = 1e3 * float(z[k + "barrier"])
    return out, mean


def match(cij, neb):
    """Members present in both files, aligned by index: (indices, cij rows, barriers)."""
    idx, ic, ib = np.intersect1d(cij[0], neb[0], return_indices=True)
    return idx, cij[1][ic], neb[1][ib]


def correlation_table(defect, data, ref, mean):
    """One line per (committee, quantity): n, x and barrier statistics, Pearson r, Spearman ρ, least-squares slope."""
    lines = [f"{defect} migration barrier of every POPS committee member against its elastic constants (7_cij_vs_barrier.py): "
             f"x from cij_members.csv (stress–strain at the member's own a0), barrier from neb_pops.npz (member's "
             f"CI-NEB from the mean band, cell at A0); r Pearson, rho Spearman; slope in meV per unit of x",
             f"{'committee':11s} {'x':4s} {'n':>3s} {'x mean':>9s} {'x std':>8s} {'E_m mean':>9s} {'E_m std':>7s} "
             f"{'r':>7s} {'rho':>7s} {'slope':>10s}"]
    for name in NAMES:
        idx, rows, b = data[name]
        for q, _, _, col in QUANT:
            x = rows[:, col]
            s, _ = linfit(x, b)
            lines.append(f"{LABEL[name]:11s} {q:4s} {len(idx):3d} {x.mean():9.4f} {x.std():8.4f} {b.mean():9.1f} {b.std():7.1f} "
                         f"{pearson(x, b):+7.3f} {spearman(x, b):+7.3f} {s:10.3g}")
        r = ref[name]
        lines.append(f"{LABEL[name]:11s} reference model: a0 {r[2]:.4f} Å, C11 {r[3]:.1f}, C12 {r[4]:.1f}, C44 {r[5]:.1f}, "
                     f"B {r[6]:.1f} GPa; barrier {mean[name]:.1f} meV")
    lines.append("(with n = 100 members, |r| above about 0.2 is significant at the 5 % level; E_m in meV)")
    return lines


def figure(defect, data, ref, mean, stem):
    """One row per committee in ROWS (its own x-axes), one column per quantity; writes stem.pdf and stem.png."""
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 9,
                         "ytick.labelsize": 10, "legend.fontsize": 10})
    window = "constrained" in ROWS                               # the ±0.1 % windows are only visible on that row's axes
    fig, axes = plt.subplots(len(ROWS), len(PANELS), figsize=(W, (0.36 if len(ROWS) == 1 else 0.66) * W),
                             sharey="row", squeeze=False)
    for i, name in enumerate(ROWS):
        idx, rows, b = data[name]
        for ax, (q, lab, target, col) in zip(axes[i], PANELS):
            x = rows[:, col]
            if q != "a0" and window:
                ax.axvspan(target * (1 - TOL), target * (1 + TOL), color="k", alpha=0.15, lw=0)
            ax.axvline(target, color="k", lw=0.7)
            s, c = linfit(x, b)
            xx = np.array([x.min(), x.max()])
            ax.plot(xx, s * xx + c, "--", lw=1.0, color=COL[name])
            ax.plot(x, b, ".", ms=4.5, alpha=0.7, color=COL[name])
            ax.plot([ref[name][col]], [mean[name]], marker="*", ms=12, color=COL[name], mec="k", mew=0.6, ls="none")
            lo, hi = min(x.min(), target), max(x.max(), target)
            pad = 0.08 * (hi - lo) or 1e-4 * abs(target)
            ax.set_xlim(lo - pad, hi + pad)
            ax.text(0.04, 0.96, f"r = {pearson(x, b):+.2f}\nρ = {spearman(x, b):+.2f}", transform=ax.transAxes, va="top",
                    fontsize=10, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
            ax.ticklabel_format(axis="x", useOffset=False)
            if i == 0 and len(ROWS) > 1:
                ax.set_title(lab)
            if i == len(ROWS) - 1:
                ax.set_xlabel(lab)
        axes[i, 0].set_ylabel(f"{LABEL[name]}: $E_\\mathrm{{m}}$ (meV)" if len(ROWS) > 1 else "$E_\\mathrm{m}$ (meV)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], ls="none", marker=".", ms=7, color=COL[ROWS[0]] if len(ROWS) == 1 else "0.4"),
               Line2D([], [], ls="none", marker="*", ms=11, color=COL[ROWS[0]] if len(ROWS) == 1 else "0.4", mec="k"),
               Line2D([], [], ls="--", color=COL[ROWS[0]] if len(ROWS) == 1 else "0.4"), Line2D([], [], color="k", lw=0.7)]
    labels = [f"{LABEL[ROWS[0]]} committee member" if len(ROWS) == 1 else "committee member", "reference model",
              "least-squares fit", "target"]
    if window:
        handles.append(Patch(fc="k", alpha=0.15)); labels.append("±0.1 % constraint window")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels) if len(ROWS) == 1 else 3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=9, handletextpad=0.4, columnspacing=1.4, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.88 if len(ROWS) == 1 else 0.92))
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def main():
    root = os.environ.get("OUT", os.path.join(os.path.dirname(HERE), "repro", "saved"))
    src = os.environ.get("REPLOT_FROM", os.path.join(root, "mace"))
    out = os.path.join(root, "mace", "results")
    os.makedirs(out, exist_ok=True)
    cij_file = os.path.join(src, "pops", "cij_members.csv")
    if not os.path.exists(cij_file):                                # as 6_neb_pops.py: the published committees suffice
        cij_file = os.path.join(os.path.dirname(HERE), "published", "mace", "pops", "cij_members.csv")
    neb_file = os.path.join(src, "results", "neb_pops.npz")
    if not os.path.exists(neb_file):
        sys.exit(f"missing {neb_file}: run mace/6_neb_pops.py (and --collect) first")
    cij, ref = load_cij(cij_file)
    print(f"C_ij from {cij_file}\nbarriers from {neb_file}")
    for defect in os.environ.get("DEFECTS", "vacancy").split(","):
        neb, mean = load_barriers(neb_file, defect)
        data = {n: match(cij[n], neb[n]) for n in NAMES}
        for n in NAMES:
            print(f"  {defect}/{LABEL[n]}: {len(data[n][0])} members matched ({len(cij[n][0])} with C_ij, {len(neb[n][0])} with a barrier)")
        lines = correlation_table(defect, data, ref, mean)
        print("\n".join(lines))
        stem = os.path.join(out, f"cij_vs_barrier_{defect}")
        open(stem + ".txt", "w").write("\n".join(lines) + "\n")
        figure(defect, data, ref, mean, stem)
        print(f"→ {stem}.pdf/.png/.txt")


if __name__ == "__main__":
    main()
