"""mace/7_cij_vs_barrier.py — the correlation measures, the two loaders and member matching, the table and the figure, on
synthetic inputs shaped like cij_members.csv (p4) and neb_pops.npz (6 --collect).  The script has a __main__ guard, so it is
imported whole (importlib: its file name starts with a digit); it needs neither torch nor the model."""
import importlib.util
import os
import tempfile
import unittest

import numpy as np

import harness
from harness import tested

F = "mace/7_cij_vs_barrier.py"
_spec = importlib.util.spec_from_file_location("cij_vs_barrier", os.path.join(harness.REPO, F))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def synthetic(d, n=40, seed=0, drop=(3, 17)):
    """cij_members.csv as p4 writes it and neb_pops.npz as 6 --collect writes it, in directory d.  Members `drop` have no NEB
    result; the unconstrained barrier is an exact affine function of C44 (r = 1), the constrained one is noise."""
    rng = np.random.default_rng(seed)
    rows, cij = [], {}
    for j, name in enumerate(("constrained", "unconstrained")):
        s = (0.01, 0.001, 0.001, 0.0) if j == 0 else (9.0, 4.0, 5.0, 0.01)
        C11, C12, C44 = (116.3 + s[0] * rng.standard_normal(n), 64.8 + s[1] * rng.standard_normal(n),
                         30.9 + s[2] * rng.standard_normal(n))
        a0 = 4.032 + s[3] * rng.standard_normal(n)
        cij[name] = np.column_stack([np.full(n, j), np.arange(n), a0, C11, C12, C44, (C11 + 2 * C12) / 3])
        rows.append([j, -1, 4.032, 116.3, 64.8, 30.9, (116.3 + 2 * 64.8) / 3])
        rows += cij[name].tolist()
    f_cij = os.path.join(d, "cij_members.csv")
    np.savetxt(f_cij, np.array(rows), delimiter=",", header="committee,member,a0,C11,C12,C44,B")
    keep = np.array([i for i in range(n) if i not in drop])
    bar = {"unconstrained": 0.5 + 0.01 * (cij["unconstrained"][keep, 5] - 30.9),            # eV, exact line in C44
           "constrained": 0.6 + 0.02 * rng.standard_normal(len(keep))}
    z = {}
    for name in ("unconstrained", "constrained"):
        z[f"vacancy_{name}_members"], z[f"vacancy_{name}_barrier_relaxed"], z[f"vacancy_{name}_barrier"] = keep, bar[name], 0.55
    f_neb = os.path.join(d, "neb_pops.npz")
    np.savez(f_neb, **z)
    return f_cij, f_neb, cij, keep, bar


class TestStatistics(unittest.TestCase):
    def test_pearson_spearman_linfit(self):
        tested(F, "pearson", "spearman", "linfit")
        x = np.linspace(0, 1, 50)
        self.assertAlmostEqual(M.pearson(x, 2 * x + 1), 1.0, places=12)
        self.assertAlmostEqual(M.pearson(x, -x), -1.0, places=12)
        self.assertTrue(np.isnan(M.pearson(np.ones(5), x[:5])))                   # a constant has no correlation
        y = np.exp(5 * x)                                                          # monotone but far from linear
        self.assertAlmostEqual(M.spearman(x, y), 1.0, places=12)
        self.assertLess(M.pearson(x, y), 0.9)
        self.assertAlmostEqual(M.spearman(x, -y), -1.0, places=12)
        s, c = M.linfit(x, 3 * x - 2)
        self.assertAlmostEqual(s, 3.0, places=10); self.assertAlmostEqual(c, -2.0, places=10)


class TestInputs(unittest.TestCase):
    def test_load_and_match(self):
        tested(F, "load_cij", "load_barriers", "match")
        with tempfile.TemporaryDirectory() as d:
            f_cij, f_neb, cij, keep, bar = synthetic(d)
            c, ref = M.load_cij(f_cij)
            for name in ("constrained", "unconstrained"):
                np.testing.assert_array_equal(c[name][0], np.arange(40))           # the reference row (-1) is split off
                np.testing.assert_allclose(c[name][1], cij[name])
                np.testing.assert_allclose(ref[name][2:], [4.032, 116.3, 64.8, 30.9, (116.3 + 2 * 64.8) / 3])
            b, mean = M.load_barriers(f_neb, "vacancy")
            for name in ("constrained", "unconstrained"):
                np.testing.assert_array_equal(b[name][0], keep)
                np.testing.assert_allclose(b[name][1], 1e3 * bar[name])            # eV → meV
                self.assertAlmostEqual(mean[name], 550.0)
            with self.assertRaises(KeyError):
                M.load_barriers(f_neb, "interstitial")
            idx, rows, bb = M.match(c["unconstrained"], b["unconstrained"])
            np.testing.assert_array_equal(idx, keep)                               # only members present in both
            np.testing.assert_allclose(rows[:, 1], keep)                           # rows follow the member index
            np.testing.assert_allclose(bb, 1e3 * bar["unconstrained"])
            # a shuffled NEB file matches the same pairs
            z = dict(np.load(f_neb)); perm = np.random.default_rng(1).permutation(len(keep))
            shuffled = (z["vacancy_unconstrained_members"][perm], 1e3 * z["vacancy_unconstrained_barrier_relaxed"][perm])
            idx2, rows2, bb2 = M.match(c["unconstrained"], shuffled)
            np.testing.assert_array_equal(idx2, idx); np.testing.assert_allclose(rows2, rows); np.testing.assert_allclose(bb2, bb)


class TestOutputs(unittest.TestCase):
    def test_table_figure_main(self):
        tested(F, "correlation_table", "figure", "main")
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "mace", "pops")); os.makedirs(os.path.join(root, "mace", "results"))
            synthetic(os.path.join(root, "mace", "pops"))
            os.replace(os.path.join(root, "mace", "pops", "neb_pops.npz"), os.path.join(root, "mace", "results", "neb_pops.npz"))
            cij, ref = M.load_cij(os.path.join(root, "mace", "pops", "cij_members.csv"))
            neb, mean = M.load_barriers(os.path.join(root, "mace", "results", "neb_pops.npz"), "vacancy")
            data = {n: M.match(cij[n], neb[n]) for n in M.NAMES}
            lines = M.correlation_table("vacancy", data, ref, mean)
            self.assertEqual(len(lines), 2 + len(M.NAMES) * (len(M.QUANT) + 1) + 1)
            c44 = [l for l in lines if l.startswith("MACE") and l.split()[1] == "C44"][0].split()
            self.assertEqual(c44[2], "38")                                          # 40 members, 2 without a barrier
            self.assertAlmostEqual(float(c44[7]), 1.0, places=3)                    # r: the barrier is a line in C44
            self.assertAlmostEqual(float(c44[8]), 1.0, places=3)                    # rho
            self.assertAlmostEqual(float(c44[9]), 10.0, places=2)                   # slope: 0.01 eV/GPa = 10 meV/GPa
            stem = os.path.join(root, "fig")
            M.figure("vacancy", data, ref, mean, stem)
            for ext in ("pdf", "png"):
                self.assertGreater(os.path.getsize(f"{stem}.{ext}"), 1000)
            env = {k: os.environ.get(k) for k in ("OUT", "DEFECTS", "REPLOT_FROM")}
            os.environ["OUT"] = root; os.environ["DEFECTS"] = "vacancy"; os.environ.pop("REPLOT_FROM", None)
            try:
                M.main()
            finally:
                for k, v in env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            out = os.path.join(root, "mace", "results", "cij_vs_barrier_vacancy")
            self.assertEqual(open(out + ".txt").read().splitlines(), lines)
            for ext in ("pdf", "png"):
                self.assertGreater(os.path.getsize(f"{out}.{ext}"), 1000)


if __name__ == "__main__":
    unittest.main()
