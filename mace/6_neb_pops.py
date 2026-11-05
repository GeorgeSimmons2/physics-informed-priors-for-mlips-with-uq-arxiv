"""
6. Point-defect migration barriers with POPS uncertainty: MACE as trained (θ0, unconstrained committee)
against the elastic-constrained readout (θc, constrained committee), every model and every committee member at
its OWN relaxed lattice constant (CELL=relaxed, the default).  Every committee member relaxes its own cell, its own
endpoints and its own band (ENDPOINTS=member, the default; ENDPOINTS=mean keeps the mean model's endpoint geometries,
scaled to the member's cell, and moves only the interior images).  By default the first N_MEMBERS = 30 members of
each committee are run: p3_sample.jl's samplers have the prefix property, so these are the members of an N = 30 run.

Two defects in fcc Al, both built in a SIZE³ cubic supercell:
  vacancy       nearest-neighbour hop into the vacant site
  interstitial  <100> split dumbbell, translation–rotation jump to the [110] neighbour (new axis [010])

Cell protocol.  CELL=relaxed: BEFORE the defect is built, the perfect fcc cell is relaxed for the readout in use
(lib.relaxed_a0: E(a) of the 4-atom conventional cell minimised to 1e-8 Å), the SIZE³ supercell is built at that a0,
and every later relaxation (endpoints, NEB) is internal, at that fixed cell.  So the mean model sits at its own a0
and each member at its own a0_m: no residual pressure for anyone.  CELL=fixed: the old protocol, everything at the
paper's A0 (4.032 Å), written under neb_pops_fixedcell* so the two protocols' results coexist.

Per (defect, model):
  1. mean model (θ0 or θc): a0 of the mean readout, defect built at a0, relax endpoints, climbing-image NEB from an
     IDPP path → the mean band, cached as $OUT/mace/results/neb_pops/<defect>_<model>_mean.xyz (reused by later runs).
  2. every member m: write θ_m into the readout (lib._set_lin, as pops/p4), relax the perfect cell → a0_m, scale the
     mean band to a0_m (fractional coordinates kept) and re-converge the climbing-image NEB from that warm start.
     ENDPOINTS=member (default): both endpoints are re-relaxed by the member first (a few BFGS steps each, cheap next
     to the NEB) and Barrier_m is relative to the member's own initial minimum at its own a0_m.  ENDPOINTS=mean: the
     endpoints stay at the mean model's relaxed geometries (scaled to a0_m) and only the interior images move.
     Each member is checked to have relaxed to the SAME transition as the mean model (path_ok): NEB converged,
     no atom of any image further than PATH_TOL (0.5 Å) from its place in the (scaled) mean band, saddle at the
     mean's image ±1, and |ΔE| < 2 meV (the two endpoints are symmetry-equivalent for both defects).
  3. for comparison, the frozen-geometry committee on the mean band (E_m = E_frozen + D·θ_m, at the mean model's
     cell and geometry) is recorded too.

Members are independent, so split them across jobs with MEMBERS=a-b (inclusive; default all) and run
`--collect` once every member file exists.  Per-member results: neb_pops/<defect>_<model>_m<i>.npz (with the
member's a0).  With ENDPOINTS=mean everything is written under neb_pops_meanends/ and neb_pops_meanends.* instead,
so the two endpoint protocols' results coexist; the mean bands are the same in both and are copied across if only one
has them.  CELL=fixed appends _fixedcell to all of these.  The script refuses to run in a directory holding files of
the other cell protocol (files written before CELL existed count as fixed): mace/run_neb_pops.sh moves them aside.

In:  θ0 from the model; θc from $OUT/mace/results/theta_constrained_elastic.csv (3_qp.jl) or, failing that,
     from the published constrained model; committees from $OUT/mace/pops/committee_{constrained,unconstrained}.csv
     (p3_sample.jl) or from published/mace/pops/.  No DFT data, no refit: the published artefacts suffice.
Out: $OUT/mace/results/neb_pops.{png,pdf,txt,npz}, neb_pops/ (mean bands, per-member files, logs)
     (ENDPOINTS=mean: neb_pops_meanends.* and neb_pops_meanends/; CELL=fixed: *_fixedcell)
Run: python3 6_neb_pops.py                 mean bands, then members MEMBERS, then collect if complete
     python3 6_neb_pops.py --collect       table and figure from the files present
     bash mace/run_neb_pops.sh             all 100 members, relaxed cells (moves old fixed-cell results aside first)
Env: OUT, MODEL_FILE, A0 (4.032, CELL=fixed only), THREADS, SIZE (3), NIMAGES (7), FMAX (5e-3), DELTA (0.85),
     DEFECTS (vacancy,interstitial), N_MEMBERS (30: the first N of each committee), MEMBERS (all | a-b, within N),
     MAX_STEPS (3000, per member NEB), PATH_TOL (0.5 Å),
     ENDPOINTS (member: each member relaxes its own endpoints | mean: members share the mean model's),
     CELL (relaxed: every model and member at its own a0 | fixed: everything at A0)
"""
import os
import shutil
import sys
import time

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.io import read, write
from ase.optimize import BFGS, FIRE

try:                                   # ASE >= 3.23
    from ase.mep import NEB
except ImportError:
    from ase.neb import NEB

import lib

SIZE = int(os.environ.get("SIZE", 3))
NIMAGES = int(os.environ.get("NIMAGES", 7))
FMAX = float(os.environ.get("FMAX", 5e-3))
DELTA = float(os.environ.get("DELTA", 0.85))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 3000))
DEFECTS = os.environ.get("DEFECTS", "vacancy,interstitial").split(",")
N_MEMBERS = int(os.environ.get("N_MEMBERS", 30))
MEMBERS = os.environ.get("MEMBERS", "all")
PATH_TOL = float(os.environ.get("PATH_TOL", 0.5))
END_MODE = os.environ.get("ENDPOINTS", "member")         # member: each member relaxes its own endpoints
assert END_MODE in ("mean", "member"), f"ENDPOINTS={END_MODE!r}: mean or member"
CELL = os.environ.get("CELL", "relaxed")                 # relaxed: every model and member at its own a0 | fixed: all at A0
assert CELL in ("relaxed", "fixed"), f"CELL={CELL!r}: relaxed or fixed"
COLLECT = "--collect" in sys.argv
PUB = os.path.join(lib.PUBLISHED, "pops")
SUFFIX = "" if CELL == "relaxed" else "_fixedcell"
STEM = ("neb_pops" if END_MODE == "member" else "neb_pops_meanends") + SUFFIX
WORK = os.path.join(lib.OUT, STEM); os.makedirs(WORK, exist_ok=True)
OTHER_WORK = os.path.join(lib.OUT, ("neb_pops_meanends" if END_MODE == "member" else "neb_pops") + SUFFIX)
MODELS = ("unconstrained", "constrained")
LABEL = {"unconstrained": "MACE", "constrained": "constrained"}
COL = {"unconstrained": "C1", "constrained": "C0"}


# ── readout weights and committees ─────────────────────────────────────────
def load_thetas():
    theta0 = lib.Basis(lib.load_calc()).theta0
    f = os.path.join(lib.OUT, "theta_constrained_elastic.csv")
    if os.path.exists(f):
        thetac, src = np.loadtxt(f, delimiter=",").ravel(), f
    else:                              # the published constrained checkpoint carries θc in its readout
        from mace.calculators import MACECalculator
        src = os.path.join(lib.PUBLISHED, "results", "Al_small_stagetwo_elastic.model")
        thetac = lib.Basis(MACECalculator(model_paths=src, device="cpu", default_dtype="float64")).theta0
    com = {}
    for name in MODELS:
        for d in (lib.POPS, PUB):
            f = os.path.join(d, f"committee_{name}.csv")
            if os.path.exists(f):
                com[name] = (np.loadtxt(f, delimiter=","), f)
                break
        else:
            raise FileNotFoundError(f"committee_{name}.csv in {lib.POPS} or {PUB}")
    print(f"θ0 from {lib.MODEL_FILE}\nθc from {src}")
    for name in MODELS:
        print(f"{name}: the first {min(N_MEMBERS, com[name][0].shape[0])} of {com[name][0].shape[0]} members from {com[name][1]}")
    return {"unconstrained": theta0, "constrained": thetac}, {n: com[n][0][:N_MEMBERS] for n in MODELS}


def set_theta(calc, theta):
    """Write θ into calc's readout weights in place (as pops/p4_evaluate.py) and drop ASE's result cache."""
    m = calc.models[0]
    ro = m.readouts
    n0, n1 = ro[0].linear.irreps_in.count("0e"), ro[1].linear_2.irreps_in.dim
    scale = float(m.scale_shift.scale.reshape(-1)[0])
    lib._set_lin(ro[0].linear, n0, theta[:n0] / scale)
    lib._set_lin(ro[1].linear_2, n1, theta[n0:n0 + n1] / scale)
    with torch.no_grad():
        m.scale_shift.shift.fill_(float(theta[-1]))
    calc.results, calc.atoms = {}, None


# ── lattice constant and defects at a ──────────────────────────────────────
def lattice_constant(calc):
    """The lattice constant the defect cells are built at: the current readout's own relaxed a0 (CELL=relaxed) —
    the perfect fcc cell is relaxed BEFORE the defect is created — or the paper's A0 (CELL=fixed)."""
    if CELL == "fixed":
        return lib.A0
    a, _ = lib.relaxed_a0(calc)
    return float(a)


def rescale(images, a_from, a_to):
    """Copies of images with the cell scaled a_from → a_to, fractional coordinates kept (no calculator attached)."""
    out = []
    for im in images:
        x = im.copy()
        if a_to != a_from:
            x.set_cell(im.cell.array * (a_to / a_from), scale_atoms=True)
        out.append(x)
    return out


def supercell(a):
    return bulk("Al", "fcc", a=a, cubic=True).repeat((SIZE, SIZE, SIZE))


def vacancy_endpoints(a):
    """Initial: vacancy at the atom nearest the cell centre. Final: a nearest neighbour has hopped into it."""
    perfect = supercell(a)
    pos = perfect.positions
    v = int(np.argmin(np.linalg.norm(pos - perfect.cell.sum(0) / 2, axis=1)))
    d = perfect.get_distances(v, range(len(perfect)), mic=True); d[v] = np.inf
    j = int(np.argmin(d))
    vec = perfect.get_distance(j, v, mic=True, vector=True)
    initial = perfect.copy(); del initial[v]
    final = initial.copy(); final.positions[j if j < v else j - 1] += vec
    return initial, final


def interstitial_endpoints(a):
    """<100> dumbbell at site A → [010] dumbbell at the [110] neighbour B (translation–rotation jump).
    Atom bookkeeping:  A+δx → B−δy (arriving),  atom at B → B+δy (pushed),  A−δx → A (back on site)."""
    perfect = supercell(a)
    pos = perfect.positions
    A = int(np.argmin(np.linalg.norm(pos - perfect.cell.sum(0) / 2, axis=1)))
    vecs = perfect.get_distances(A, range(len(perfect)), mic=True, vector=True)
    target = 0.5 * a * np.array([1.0, 1.0, 0.0])
    dev = np.linalg.norm(vecs - target, axis=1); dev[A] = np.inf
    B = int(np.argmin(dev)); assert dev[B] < 1e-6
    rA, rB = pos[A].copy(), pos[A] + target
    x, y = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    others = [i for i in range(len(perfect)) if i not in (A, B)]
    ini = np.vstack([pos[others], [rB], [rA + DELTA * x], [rA - DELTA * x]])
    fin = np.vstack([pos[others], [rB + DELTA * y], [rB - DELTA * y], [rA]])
    mk = lambda p: Atoms(["Al"] * len(p), positions=p, cell=perfect.cell, pbc=True)
    return mk(ini), mk(fin)


ENDPOINTS = {"vacancy": vacancy_endpoints, "interstitial": interstitial_endpoints}


def dumbbell_ok(defect, images):
    """Interstitial endpoints must still be [100] / [010] dumbbells after relaxation."""
    if defect != "interstitial":
        return True
    n = len(images[0])
    axes = [images[0].get_distance(n - 2, n - 1, mic=True, vector=True),
            images[-1].get_distance(n - 3, n - 2, mic=True, vector=True)]
    axes = [np.abs(a / np.linalg.norm(a)) for a in axes]
    return bool(axes[0][0] > 0.9 and axes[1][1] > 0.9)


def path_frac(images):
    s = np.zeros(len(images))
    for i in range(1, len(images)):
        d = images[i].positions - images[i - 1].positions
        d -= np.round(d @ np.linalg.inv(images[i].cell.array)) @ images[i].cell.array
        s[i] = s[i - 1] + np.linalg.norm(d)
    return s / s[-1], s[-1]


def converge_band(images, calc, log, warm, relax_ends=True):
    """Climbing-image NEB; relax_ends=True relaxes both endpoints first. warm=False builds the IDPP path first."""
    for im in images:
        im.calc = calc
    if relax_ends:
        for at in (images[0], images[-1]):
            BFGS(at, logfile=log).run(fmax=FMAX, steps=5000)
    neb = NEB(images, climb=True, k=0.1, allow_shared_calculator=True)
    if not warm:
        neb.interpolate(method="idpp")
    opt = FIRE(neb, logfile=log)
    opt.run(fmax=FMAX, steps=MAX_STEPS if warm else 20000)
    E = np.array([im.get_potential_energy() for im in images])
    fmax_band = float(np.sqrt((neb.get_forces() ** 2).sum(1).max()))
    return E, fmax_band, opt.nsteps


def summarize(E, images):
    rel = E - E[0]
    frac, length = path_frac(images)
    saddle = int(np.argmax(rel))
    return dict(rel=rel, frac=frac, length=length, saddle=saddle, barrier=rel[saddle], dE=rel[-1])


def path_dev(pos, pos_mean, cell):
    """Largest displacement (Å, minimum image) of any atom of any image from its place in the mean band."""
    d = np.asarray(pos) - np.asarray(pos_mean)
    inv = np.linalg.inv(np.asarray(cell))
    d -= np.round(d @ inv) @ np.asarray(cell)
    return float(np.sqrt((d ** 2).sum(-1)).max())


def path_check(dev, saddle, saddle_mean, dE, converged):
    """Did the member relax to the mean model's transition?  Same path (dev < PATH_TOL), saddle at the same image ±1,
    symmetric endpoints (|ΔE| < 2 meV; both defects' endpoints are symmetry-equivalent), NEB converged."""
    return bool(converged and dev < PATH_TOL and abs(int(saddle) - int(saddle_mean)) <= 1 and abs(dE) < 2e-3)


# ── mean band (cached) and members ─────────────────────────────────────────
def cell_mode(npz):
    """The CELL protocol a result file was written under (files from before CELL existed are fixed-cell)."""
    z = np.load(npz, allow_pickle=False)
    return str(z["cell_mode"]) if "cell_mode" in z.files else "fixed"


def check_work():
    """Refuse to mix cell protocols: every .npz in WORK must have been written with the current CELL."""
    stale = [fn for fn in sorted(os.listdir(WORK)) if fn.endswith(".npz") and cell_mode(os.path.join(WORK, fn)) != CELL]
    if stale:
        other = "fixed" if CELL == "relaxed" else "relaxed"
        sys.exit(f"{len(stale)} file(s) in {WORK} were written with CELL={other} (e.g. {stale[0]}, {stale[-1]}); the "
                 f"current run is CELL={CELL}.  Move them aside first (mace/run_neb_pops.sh moves fixed-cell results to "
                 f"{WORK}_fixedcell/) or run with CELL={other}.")


def mean_band(defect, name, calc, theta_mean, log):
    f = os.path.join(WORK, f"{defect}_{name}_mean")
    other = os.path.join(OTHER_WORK, f"{defect}_{name}_mean")
    if (not os.path.exists(f + ".npz") and os.path.exists(other + ".xyz") and os.path.exists(other + ".npz")
            and cell_mode(other + ".npz") == CELL):
        for ext in (".xyz", ".npz"):   # the mean band does not depend on ENDPOINTS: reuse the other protocol's
            shutil.copy2(other + ext, f + ext)
    if os.path.exists(f + ".xyz") and os.path.exists(f + ".npz"):
        z = np.load(f + ".npz")
        a0 = float(z["a0"]) if "a0" in z.files else lib.A0
        images = read(f + ".xyz", ":")
        for im in images:
            im.calc = calc
        print(f"  {defect}/{LABEL[name]}: mean band from cache (a0 = {a0:.4f} Å)")
        return images, int(z["saddle"]), a0
    t = time.time()
    a0 = lattice_constant(calc)        # relax the perfect cell first, then build the defect in it
    initial, final = ENDPOINTS[defect](a0)
    images = [initial] + [initial.copy() for _ in range(NIMAGES - 2)] + [final]
    E, fmax_band, nsteps = converge_band(images, calc, log, warm=False)
    s = summarize(E, images)
    # frozen-geometry committee needs D on the band; E_frozen = E_mean − D·θ_mean
    basis = lib.Basis(calc)
    assert np.allclose(basis.theta0, theta_mean, atol=1e-8), "calculator readout ≠ θ_mean"
    D = np.array([basis.row(im)[0] for im in images])
    write(f + ".xyz", images)
    np.savez(f + ".npz", E=E, D=D, E_frozen=E - D @ theta_mean, fmax_band=fmax_band, nsteps=nsteps,
             dumbbell_ok=dumbbell_ok(defect, images), a0=a0, cell_mode=CELL, **{k: v for k, v in s.items()})
    print(f"  {defect:12s} {LABEL[name]:11s} mean: a0 {a0:.4f} Å, barrier {1e3 * s['barrier']:7.1f} meV, "
          f"ΔE {1e3 * s['dE']:+.2f} meV, |F|max {fmax_band:.3f}, {nsteps} NEB steps, {len(images[0])} atoms, "
          f"{time.time() - t:.0f} s", flush=True)
    return images, s["saddle"], a0


def member(defect, name, i, theta, images_mean, saddle_mean, a_mean, calc, log):
    f = os.path.join(WORK, f"{defect}_{name}_m{i}.npz")
    if os.path.exists(f):
        return
    t = time.time()
    set_theta(calc, theta)
    a = lattice_constant(calc)         # the member's own relaxed cell, before anything else moves
    images = rescale(images_mean, a_mean, a)
    E, fmax_band, nsteps = converge_band(images, calc, log, warm=True, relax_ends=END_MODE == "member")
    s = summarize(E, images)
    ok = dumbbell_ok(defect, images)
    converged = nsteps < MAX_STEPS
    pos = np.array([im.positions for im in images])
    pos_ref = [im.positions for im in rescale(images_mean, a_mean, a)]
    dev = path_dev(pos, pos_ref, images[0].cell.array)
    path_ok = path_check(dev, s["saddle"], saddle_mean, s["dE"], converged)
    np.savez(f, E=E, fmax_band=fmax_band, nsteps=nsteps, converged=converged, dumbbell_ok=ok, positions=pos,
             path_dev=dev, path_ok=path_ok, saddle_mean=saddle_mean, a0=a, a0_mean=a_mean, cell_mode=CELL,
             **{k: v for k, v in s.items()})
    print(f"    m{i:<3d} a0 {a:.4f}  barrier {1e3 * s['barrier']:7.1f} meV  ΔE {1e3 * s['dE']:+6.2f}  |F|max {fmax_band:.3f}  "
          f"{nsteps:3d} steps  dev {dev:.3f} Å{'' if converged else '  NOT CONVERGED'}{'' if ok else '  dumbbell lost'}"
          f"{'' if path_ok else '  PATH DIFFERS FROM MEAN'}  {time.time() - t:.0f} s", flush=True)


def member_range(n):
    if MEMBERS == "all":
        return range(n)
    a, b = (int(x) for x in MEMBERS.split("-"))
    return range(a, min(b, n - 1) + 1)


# ── collect ────────────────────────────────────────────────────────────────
def collect(thetas, com):
    res, missing = {}, []
    for defect in DEFECTS:
        for name in MODELS:
            m = np.load(os.path.join(WORK, f"{defect}_{name}_mean.npz"))
            pos_mean = [im.positions for im in read(os.path.join(WORK, f"{defect}_{name}_mean.xyz"), ":")]
            cell = read(os.path.join(WORK, f"{defect}_{name}_mean.xyz")).cell.array
            N = len(com[name])
            files = [os.path.join(WORK, f"{defect}_{name}_m{i}.npz") for i in range(N)]
            have = [i for i, f in enumerate(files) if os.path.exists(f)]
            missing += [f"{defect}/{name}/m{i}" for i in range(N) if i not in have]
            mem = [dict(np.load(files[i])) for i in have]
            a_mean = float(m["a0"]) if "a0" in m.files else lib.A0
            a_mem = np.array([float(z["a0"]) if "a0" in z else lib.A0 for z in mem])
            for z, a in zip(mem, a_mem):   # files written before the path check carry positions: check them now
                if "path_dev" not in z:
                    z["path_dev"] = path_dev(z["positions"], np.asarray(pos_mean) * (a / a_mean), cell * (a / a_mean))
                    z["path_ok"] = path_check(z["path_dev"], z["saddle"], m["saddle"], z["dE"], z["converged"])
            frac = m["frac"]
            rel_relaxed = np.array([np.interp(frac, z["frac"], z["rel"]) for z in mem]) if mem else np.zeros((0, NIMAGES))
            # frozen-geometry committee on the mean band, for comparison
            E_fro = m["E_frozen"][None, :] + com[name] @ m["D"].T
            rel_frozen = E_fro - E_fro[:, :1]
            res[defect, name] = dict(
                frac=frac, rel_mean=m["rel"], barrier=float(m["barrier"]), dE=float(m["dE"]),
                fmax_band=float(m["fmax_band"]), members=np.array(have), a0=a_mean, a0_relaxed=a_mem,
                barrier_relaxed=np.array([float(z["barrier"]) for z in mem]),
                dE_relaxed=np.array([float(z["dE"]) for z in mem]),
                converged=np.array([bool(z["converged"]) for z in mem]),
                dumbbell_ok=np.array([bool(z["dumbbell_ok"]) for z in mem]),
                path_dev=np.array([float(z["path_dev"]) for z in mem]),
                path_ok=np.array([bool(z["path_ok"]) for z in mem]),
                rel_relaxed=rel_relaxed, rel_frozen=rel_frozen,
                barrier_frozen=rel_frozen[:, int(m["saddle"])])
    if missing:
        print(f"{len(missing)} member files missing (e.g. {', '.join(missing[:5])}); collecting what is present")

    how = ("re-relaxes endpoints and band" if END_MODE == "member" else
           "re-converges the band between the mean model's relaxed endpoints") + " from the mean model's path"
    where = (f"{SIZE}³ cubic cell, every model and member at its OWN relaxed a0 (perfect cell relaxed before the defect "
             f"is built; endpoints and band relaxed internally at that cell)" if CELL == "relaxed" else
             f"{SIZE}³ cubic cell held at A0 = {lib.A0:.4f} Å for every model and member")
    lines = [f"migration barriers, {where}, {NIMAGES}-image CI-NEB; the first {N_MEMBERS} "
             f"members of each committee; every member {how} (FMAX {FMAX}); 'frozen' = members on the mean band",
             f"{'defect':12s} {'model':11s} {'mean':>7s} {'relaxed mean±σ':>16s} {'min–max':>15s} {'frozen mean±σ':>16s} "
             f"{'n':>4s} {'conv':>4s} {'ok':>4s} {'path':>4s} {'dev max':>7s} {'a0 mean':>8s} {'a0 members (Å)':>17s}"]
    for (defect, name), r in res.items():
        b, bf, a = 1e3 * r["barrier_relaxed"], 1e3 * r["barrier_frozen"], r["a0_relaxed"]
        lines.append(f"{defect:12s} {LABEL[name]:11s} {1e3 * r['barrier']:7.1f} {b.mean():8.1f} ± {b.std():5.1f} "
                     f"{b.min():7.1f}–{b.max():6.1f} {bf.mean():8.1f} ± {bf.std():5.1f} {len(b):4d} "
                     f"{r['converged'].sum():4d} {r['dumbbell_ok'].sum():4d} {r['path_ok'].sum():4d} "
                     f"{r['path_dev'].max() if len(b) else 0:7.3f} {r['a0']:8.4f} "
                     f"{a.mean() if len(a) else 0:8.4f} ± {a.std() if len(a) else 0:6.4f}")
        bad = r["members"][~r["path_ok"]]
        if len(bad):
            lines.append(f"{'':24s} path differs from the mean model's for member(s) {', '.join(map(str, bad))}")
    for defect in DEFECTS:
        u, c = res[defect, "unconstrained"], res[defect, "constrained"]
        lines.append(f"{defect:12s} constrained − MACE: {1e3 * (c['barrier'] - u['barrier']):+.1f} meV; relaxed σ "
                     f"MACE {1e3 * u['barrier_relaxed'].std():.1f}, constrained {1e3 * c['barrier_relaxed'].std():.1f} meV")
    lines.append("(mean: the mean model's barrier; n: members collected; conv: NEB converged; ok: endpoints kept the "
                 "expected defect, always true for the vacancy; path: member relaxed to the mean model's transition — "
                 f"no atom moved more than PATH_TOL = {PATH_TOL} Å from the mean band (dev max: the largest such move, Å), "
                 "saddle at the same image ±1, |ΔE| < 2 meV, converged; a0: the lattice constant the mean model's and the "
                 "members' cells were built at; the frozen committee is evaluated on the mean band at the mean model's "
                 "cell; energies meV)")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(lib.OUT, STEM + ".txt"), "w").write(txt + "\n")
    np.savez(os.path.join(lib.OUT, STEM + ".npz"), A0=lib.A0, size=SIZE, endpoints=END_MODE, cell=CELL,
             n_members=N_MEMBERS, path_tol=PATH_TOL,
             **{f"{d}_{n}_{k}": v for (d, n), r in res.items() for k, v in r.items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10})
    fig, axes = plt.subplots(1, len(DEFECTS), figsize=(3.54 * len(DEFECTS), 3.4), squeeze=False)
    for ax, defect, tag in zip(axes[0], DEFECTS, "abcd"):
        for name in MODELS:
            r = res[defect, name]
            x, M = r["frac"], 1e3 * r["rel_relaxed"]
            if len(M):
                ax.fill_between(x, M.min(0), M.max(0), color=COL[name], alpha=0.12, lw=0)
                ax.fill_between(x, M.mean(0) - M.std(0), M.mean(0) + M.std(0), color=COL[name], alpha=0.3, lw=0)
            b = 1e3 * r["barrier_relaxed"]
            ax.plot(x, 1e3 * r["rel_mean"], "-o", color=COL[name], ms=3.5, lw=1.5,
                    label=f"{LABEL[name]}: {1e3 * r['barrier']:.0f} ± {b.std() if len(b) else 0:.0f} meV")
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xlim(0, 1)
        ax.set_xlabel("fractional path length"); ax.set_ylabel(r"$E - E_\mathrm{initial}$ (meV)")
        ax.set_title(f"({tag}) {defect} migration", loc="left")
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(lib.OUT, f"{STEM}.{ext}"), dpi=200)
    print(f"→ {os.path.join(lib.OUT, STEM)}.pdf/.png/.txt/.npz")


def main():
    thetas, com = load_thetas()
    check_work()
    if COLLECT:
        collect(thetas, com)
        return
    calcs = {"unconstrained": lib.load_calc(), "constrained": lib.patched_calc(thetas["constrained"])}
    for name, c in calcs.items():
        a, _ = lib.relaxed_a0(c)
        built = "its own relaxed a0, as is every member" if CELL == "relaxed" else f"A0 = {lib.A0:.4f} Å"
        print(f"{LABEL[name]:11s} relaxed a0 {a:.4f} Å (cells built at {built})")
    print(f"CELL={CELL} ENDPOINTS={END_MODE}: members "
          f"{'relax their own endpoints' if END_MODE == 'member' else 'share the mean model endpoints'}; files under {WORK}")
    log = os.path.join(WORK, f"opt_{MEMBERS}.log")
    open(log, "w").close()
    for defect in DEFECTS:
        for name in MODELS:
            calc = calcs[name]
            set_theta(calc, thetas[name])
            images_mean, saddle_mean, a_mean = mean_band(defect, name, calc, thetas[name], log)
            rng = member_range(len(com[name]))
            print(f"  {defect}/{LABEL[name]}: members {rng.start}–{rng.stop - 1}", flush=True)
            for i in rng:
                member(defect, name, i, com[name][i], images_mean, saddle_mean, a_mean, calc, log)
            set_theta(calc, thetas[name])
    complete = all(os.path.exists(os.path.join(WORK, f"{d}_{n}_m{i}.npz"))
                   for d in DEFECTS for n in MODELS for i in range(len(com[n])))
    if complete:
        collect(thetas, com)
    else:
        print("members from other jobs still missing; run with --collect when they are done")


if __name__ == "__main__":
    main()
