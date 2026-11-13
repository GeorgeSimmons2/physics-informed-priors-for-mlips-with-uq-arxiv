"""
Compare this run's results with the published ones: every .csv and .npz that both results/ and published/ hold,
number by number, and every figure pixel by pixel.

Run:  python scripts/check_against_published.py
Read the output knowing what can differ.  Stages that only plot or sample are exact.  The constrained fits stop at
a solver tolerance, and their inputs carry round-off that depends on the machine and the thread counts, so a rerun
on other hardware agrees with the published parameters to the solver tolerance rather than to the last bit.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, PUB = os.environ.get("OUT", os.path.join(ROOT, "results")), os.path.join(ROOT, "published")


def table(path):
    """A CSV as an array of floats (a header line or a column of labels becomes nan)."""
    return np.genfromtxt(path, delimiter=",", comments="#")


def compare(a, b):
    if a.shape != b.shape:
        return f"shape {a.shape}, published {b.shape}"
    both = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a - b)[both]
    if d.size == 0 or d.max() == 0:
        return "identical"
    return f"max |Δ| {d.max():.2e}, relative to the largest value {d.max() / np.abs(b[both]).max():.1e}"


n = 0
for folder, _, files in sorted(os.walk(RES)):
    for f in sorted(files):
        new, rel = os.path.join(folder, f), os.path.relpath(os.path.join(folder, f), RES)
        pub = os.path.join(PUB, rel)
        if not os.path.exists(pub):
            continue
        if f.endswith(".csv"):
            out = compare(table(new), table(pub))
        elif f.endswith(".npz"):
            x, y = np.load(new), np.load(pub)
            out = "; ".join(sorted({compare(np.asarray(x[k], float), np.asarray(y[k], float)) for k in y.files if k in x.files}))
        elif f.endswith(".png"):
            from PIL import Image
            x, y = np.asarray(Image.open(new).convert("RGBA")), np.asarray(Image.open(pub).convert("RGBA"))
            out = "identical" if x.shape == y.shape and (x == y).all() else (
                f"{100 * np.mean(np.any(x != y, axis=2)):.3f} % of pixels differ" if x.shape == y.shape else "another size")
        else:
            continue
        n += 1
        print(f"{rel:44s} {out}")
print(f"{n} files compared" if n else f"nothing to compare: no results under {RES}")
sys.exit(0)
