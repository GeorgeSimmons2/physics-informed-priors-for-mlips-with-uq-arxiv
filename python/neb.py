"""
Migration barriers in FCC Al by climbing-image NEB (Figure 4).

Two defects in a 3×3×3 cubic supercell:
    vacancy       a nearest neighbour hops into the vacant site
    interstitial  a <100> split dumbbell jumps to the [110] neighbour site and turns to [010]

Every model sits in its OWN relaxed cell: the perfect cell is relaxed for the readout weights in use before the
defect is built, and the endpoints and the band are then relaxed at that fixed cell.  The mean model's band starts
from an IDPP path; a committee member starts from the mean model's band scaled to the member's cell, and relaxes
its own endpoints and its own band.
"""
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.optimize import BFGS, FIRE

try:                                   # ASE >= 3.23
    from ase.mep import NEB
except ImportError:
    from ase.neb import NEB

import mace_readout as mr

SIZE = 3            # supercell: SIZE³ cubic cells
NIMAGES = 7         # images of the band, endpoints included
FMAX = 5e-4         # eV/Å: force tolerance of the endpoint relaxations and of the band
DELTA = 0.85        # Å: half the initial separation of the dumbbell atoms
MAX_STEPS = 3000    # NEB steps allowed to a committee member


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


def rescale(images, a_from, a_to):
    """Copies of images with the cell scaled a_from → a_to, fractional coordinates kept (no calculator attached)."""
    out = []
    for im in images:
        x = im.copy()
        if a_to != a_from:
            x.set_cell(im.cell.array * (a_to / a_from), scale_atoms=True)
        out.append(x)
    return out


def path_frac(images):
    """Fractional path length of every image along the band."""
    s = np.zeros(len(images))
    for i in range(1, len(images)):
        d = images[i].positions - images[i - 1].positions
        d -= np.round(d @ np.linalg.inv(images[i].cell.array)) @ images[i].cell.array
        s[i] = s[i - 1] + np.linalg.norm(d)
    return s / s[-1]


def converge_band(images, calc, log, warm):
    """Relax both endpoints, then the climbing-image NEB.  warm=False builds the IDPP path first.
    Returns the image energies and the number of NEB steps."""
    for im in images:
        im.calc = calc
    for at in (images[0], images[-1]):
        BFGS(at, logfile=log).run(fmax=FMAX, steps=5000)
    band = NEB(images, climb=True, k=0.1, allow_shared_calculator=True)
    if not warm:
        band.interpolate(method="idpp")
    opt = FIRE(band, logfile=log)
    opt.run(fmax=FMAX, steps=MAX_STEPS if warm else 20000)
    E = np.array([im.get_potential_energy() for im in images])
    return E, opt.nsteps


def mean_band(defect, calc, log):
    """The band of the model whose weights are in `calc`: (images, E, a0), from an IDPP start at its own a0."""
    a0, _ = mr.relaxed_a0(calc)                 # relax the perfect cell first, then build the defect in it
    initial, final = ENDPOINTS[defect](a0)
    images = [initial] + [initial.copy() for _ in range(NIMAGES - 2)] + [final]
    E, _ = converge_band(images, calc, log, warm=False)
    return images, E, a0


def member_band(theta, images_mean, a_mean, calc, log):
    """The band of a committee member, warm-started from the mean band: (images, E, a0, converged)."""
    mr.set_theta(calc, theta)
    a, _ = mr.relaxed_a0(calc)                  # the member's own relaxed cell, before anything else moves
    images = rescale(images_mean, a_mean, a)
    E, nsteps = converge_band(images, calc, log, warm=True)
    return images, E, a, nsteps < MAX_STEPS
