"""
MACE's readout as a linear model (Figures 3 and 4).

The energy of a MACE model is linear in its readout weights:  E = E_frozen + D·θ, with
    D   per-configuration sums of the invariant features of both layers, plus the atom count (81 columns here)
    θ   the first layer's linear readout weights (64), the weights of the last linear map of the second layer's
        MLP (16), and the per-atom shift (1), all in eV
`Basis` extracts D (and its position derivatives, the force rows) from a forward pass; `set_theta` writes a new θ
back into the network, so a refitted readout is again an ordinary MACE model.

Importing this module has no side effects.  Paths follow the Julia library: results go to results/ (or $OUT) and
inputs fall back to published/.
"""
import os

import numpy as np
import torch
from ase.build import bulk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.environ.get("MODEL_FILE", os.path.join(ROOT, "models", "mace", "Al_small_stagetwo.model"))
EV_A3_TO_GPA = 160.21766208
C_TARGET = {"C11": 116.3, "C12": 64.8, "C44": 30.9}    # GPa, the elastic constants of Al
A0 = 4.032                                              # Å, the lattice constant of Al near 0 K


def results(*parts):
    """A path under the results directory ($OUT, default results/); its parent directory is created."""
    path = os.path.join(os.environ.get("OUT", os.path.join(ROOT, "results")), *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def input_file(*parts):
    """An input file: this run's result if it exists, the published one otherwise."""
    for base in (os.environ.get("OUT", os.path.join(ROOT, "results")), os.path.join(ROOT, "published")):
        path = os.path.join(base, *parts)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no {os.path.join(*parts)} under results/ or published/: run the stage that writes it first")


def use_threads():
    """Torch threads from $THREADS (default: every core)."""
    torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))


def load_calc(model_file=MODEL_FILE):
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=model_file, device="cpu", default_dtype="float64")


# ── the readout weights ────────────────────────────────────────────────────
def probe(lin, n):
    """Effective weights of an e3nn 0e → 0e Linear (unit-vector probe)."""
    X = torch.zeros(n, lin.irreps_in.dim, dtype=torch.float64)
    X[:, :n] = torch.eye(n, dtype=torch.float64)
    with torch.no_grad():
        return lin(X)[:, 0].numpy()


def _set_lin(lin, n, w_eff):
    with torch.no_grad():
        lin.weight.fill_(1.0)
    alpha = probe(lin, n)
    with torch.no_grad():
        lin.weight.copy_(torch.as_tensor(w_eff / alpha, dtype=lin.weight.dtype).reshape(lin.weight.shape))


def set_theta(calc, theta):
    """Write θ into the calculator's readout weights, in place, and drop ASE's result cache."""
    m = calc.models[0]
    ro = m.readouts
    n0, n1 = ro[0].linear.irreps_in.count("0e"), ro[1].linear_2.irreps_in.dim
    assert theta.size == n0 + n1 + 1, (theta.size, n0 + n1 + 1)
    scale = float(m.scale_shift.scale.reshape(-1)[0])
    _set_lin(ro[0].linear, n0, theta[:n0] / scale)
    _set_lin(ro[1].linear_2, n1, theta[n0:n0 + n1] / scale)
    with torch.no_grad():
        m.scale_shift.shift.fill_(float(theta[-1]))
    calc.results, calc.atoms = {}, None


class Basis:
    """Hooks a calculator's model; D and E (and F, G = −∂D/∂r) from one forward pass."""

    def __init__(self, calc):
        self.calc = calc
        m = calc.models[0]
        ro = m.readouts
        assert len(ro) == 2 and hasattr(ro[1], "linear_2"), "expected Linear + NonLinear readouts"
        self.n0, self.n1 = ro[0].linear.irreps_in.count("0e"), ro[1].linear_2.irreps_in.dim
        scale = float(m.scale_shift.scale.reshape(-1)[0])
        self.theta0 = np.concatenate([scale * probe(ro[0].linear, self.n0), scale * probe(ro[1].linear_2, self.n1),
                                      [float(m.scale_shift.shift.reshape(-1)[0])]])
        self.buf, self.keep_graph = {}, False
        grab = lambda key: (lambda mod, a: self.buf.__setitem__(key, a[0] if self.keep_graph else a[0].detach()))
        ro[0].register_forward_pre_hook(grab("h0"))
        ro[1].linear_2.register_forward_pre_hook(grab("x1"))

    def _D(self):
        return torch.cat([self.buf["h0"][:, :self.n0].sum(0), self.buf["x1"].sum(0)])

    def row(self, atoms):
        """(D, E_MACE) from one energy-only forward pass (no ASE result cache)."""
        batch = self.calc._atoms_to_batch(atoms)
        with torch.no_grad():
            out = self.calc.models[0](batch.to_dict(), training=False, compute_force=False, compute_stress=False)
        return np.concatenate([self._D().numpy(), [len(atoms)]]), float(out["energy"].reshape(-1)[0])

    def D(self, atoms):
        return self.row(atoms)[0]

    def row_forces(self, atoms):
        """(D, E_MACE, F_MACE (N×3), G (3N × 81) = −∂D/∂r, rows atom-major as F.ravel())."""
        data = self.calc._atoms_to_batch(atoms).to_dict()
        pos = data["positions"]
        self.keep_graph = True
        with torch.enable_grad():
            pos.requires_grad_(True)
            out = self.calc.models[0](data, training=False, compute_force=False, compute_stress=False)
            E, Dv = out["energy"].reshape(-1)[0], self._D()
            (gE,) = torch.autograd.grad(E, pos, retain_graph=True)
            m = Dv.numel()
            try:
                (J,) = torch.autograd.grad(Dv, pos, grad_outputs=torch.eye(m, dtype=Dv.dtype), is_grads_batched=True)
            except Exception:
                J = torch.stack([torch.autograd.grad(Dv[j], pos, retain_graph=True)[0] for j in range(m)])
        self.keep_graph = False
        n = len(atoms)
        G = np.zeros((3 * n, m + 1))
        G[:, :m] = -J.detach().reshape(m, 3 * n).T.numpy()
        return np.concatenate([Dv.detach().numpy(), [n]]), float(E.detach()), -gE.detach().numpy(), G


def training_rows(basis, frames, report=None):
    """
    The regression rows of a list of DFT frames (ASE atoms with info["dft_energy"] and arrays["dft_forces"]):
        XE, yE   one row per frame:            D/N  and  (E_DFT − E_frozen)/N
        XF, yF   one row per force component:  G = −∂D/∂r  and  F_DFT − F_frozen
        GtG, Gty, fstats = [[Σ yF², n_F]]      the force rows' normal equations, summed frame by frame
    with E_frozen = E_MACE − D·θ0 and F_frozen = F_MACE − G·θ0 the parts of the model the readout does not touch.
    """
    theta0 = basis.theta0
    p = theta0.size
    XE, yE, XF, yF = [], [], [], []
    GtG, Gty, yty, nF = np.zeros((p, p)), np.zeros(p), 0.0, 0
    for i, at in enumerate(frames):
        D, Em, Fm, G = basis.row_forces(at)
        n = len(at)
        E_frozen = Em - D @ theta0
        y = (at.arrays["dft_forces"] - Fm).ravel() + G @ theta0
        XE.append(D / n); yE.append((at.info["dft_energy"] - E_frozen) / n)
        XF.append(G); yF.append(y)
        GtG += G.T @ G; Gty += G.T @ y; yty += y @ y; nF += y.size
        if report is not None:
            report(i)
    return dict(XE=np.array(XE), yE=np.array(yE), XF=np.vstack(XF), yF=np.concatenate(yF),
                GtG=GtG, Gty=Gty[None, :], fstats=np.array([[yty, nF]]))


# ── FCC cells, lattice constant and elastic constants ─────────────────────
def fcc(a):
    return bulk("Al", "fcc", a=a, cubic=True)


def strained(atoms, e1=0.0, e2=0.0, e4=0.0):
    """cell' = cell·(I + ε)ᵀ, Voigt ε₄ = 2ε_yz."""
    eps = np.array([[e1, 0, 0], [0, e2, e4 / 2], [0, e4 / 2, 0]])
    at = atoms.copy()
    at.set_cell(at.cell.array @ (np.eye(3) + eps).T, scale_atoms=True)
    return at


def relaxed_a0(calc):
    """The calculator's equilibrium lattice constant (Å) and energy per atom: E(a) of the 4-atom cell minimised."""
    from scipy.optimize import minimize_scalar

    def E(a):
        x = fcc(a)
        x.calc = calc
        return x.get_potential_energy() / len(x)
    r = minimize_scalar(E, bounds=(3.8, 4.3), method="bounded", options={"xatol": 1e-8})
    return float(r.x), float(r.fun)


def stress_strain_cij(calc):
    """a0 and C11, C12, C44, B (GPa) from MACE's analytic stress under ±0.5% strain at the model's own a0."""
    from ase.units import GPa
    a, _ = relaxed_a0(calc)
    c0, h, st = fcc(a), 5e-3, {}
    for lab, kw in (("p1", dict(e1=h)), ("m1", dict(e1=-h)), ("p4", dict(e4=h)), ("m4", dict(e4=-h))):
        y = strained(c0, **kw); y.calc = calc; st[lab] = y.get_stress()
    s1, s4 = (st["p1"] - st["m1"]) / (2 * h), (st["p4"] - st["m4"]) / (2 * h)
    C11, C12, C44 = s1[0] / GPa, s1[1] / GPa, s4[3] / GPa
    return {"a0": a, "C11": C11, "C12": C12, "C44": C44, "B": (C11 + 2 * C12) / 3}


def constraint_rows(basis, calc, h_a=1e-3, h_e=1e-4):
    """
    Derivatives of the readout basis D of the 4-atom FCC cell at A0, by central differences (h_a in the lattice
    constant, h_e in strain):  dD/da, d²D/da², d²D/dε11², d²D/dε11dε22, d²D/dε44²  — so that for any θ
        C_ij = (160.2/V) d²D/dε_ij·θ (GPa),   P ∝ dD/da·θ,
    and the same derivatives of the frozen energy E_MACE − D·θ0 (a constant, so they vanish up to round-off).
    Returns (rows, frozen): five rows of 81, and five numbers.
    """
    c0 = fcc(A0)

    def E_frozen(at):
        x = at.copy(); x.calc = calc
        return x.get_potential_energy() - basis.D(at) @ basis.theta0

    def derivs(f):
        f0, fp, fm = f(c0), f(fcc(A0 + h_a)), f(fcc(A0 - h_a))
        h = h_e
        return [(fp - fm) / (2 * h_a),
                (fp - 2 * f0 + fm) / h_a**2,
                (f(strained(c0, e1=h)) - 2 * f0 + f(strained(c0, e1=-h))) / h**2,
                (f(strained(c0, e1=h, e2=h)) - f(strained(c0, e1=h, e2=-h))
                 - f(strained(c0, e1=-h, e2=h)) + f(strained(c0, e1=-h, e2=-h))) / (4 * h**2),
                (f(strained(c0, e4=h)) - 2 * f0 + f(strained(c0, e4=-h))) / h**2]

    return derivs(basis.D), np.array(derivs(E_frozen))
