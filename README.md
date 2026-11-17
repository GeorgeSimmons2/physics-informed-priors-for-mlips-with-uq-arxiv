# Physics-informed priors for MLIPs with UQ: the figures

Everything needed to reproduce the four figures of *Physics Informed Priors for Machine Learning Interatomic
Potentials with Uncertainty Quantification* ([arXiv:2611.09325](https://arxiv.org/abs/2611.09325)), and nothing else:
the DFT data, the fitted models, the published results, a small library and one short script per stage.

| | figure | drawn by | from |
|---|---|---|---|
| 1 | Phonon bands of unconstrained and constrained POPS ensembles, ACE models with 91 and 684 parameters | `scripts/fig1_bands/plot.jl` | `fig1/al12/bands.jls`, `fig1/al16/bands.jls` |
| 2 | RLS against constrained RLS at 200 K: phonon bands and radial distribution functions | `scripts/fig2_rls_200K/4_plot.jl` | `fig2/bands_*.csv`, `fig2/*/rdf_200K.csv` |
| 3 | Lattice and elastic constants of the constrained MACE readout and its POPS committee | `scripts/fig3_mace_elastic/6_plot.py` | `fig3/cij_members.csv` |
| 4 | Vacancy and interstitial migration barriers, MACE against constrained MACE | `scripts/fig4_migration_barriers/2_plot.py` | `fig4/neb_pops.npz` |

## Quick start

```bash
bash run.sh setup     # .venv/ from requirements.txt, the Julia environment from Manifest.toml   (once, ~40 min)
bash run.sh plots     # the four figures from published/ → results/                             (minutes)
bash run.sh all       # every stage, from the fitted models → results/                          (SLURM, about a day)
bash run.sh check     # compare results/ with published/
```

Julia 1.11 and Python 3.9.  `run.sh fig1` … `fig4` submit one figure's stages.  Any single stage can also be run
by hand with the command in its header: a stage reads its inputs from `results/` and falls back to `published/`,
so it starts from the published results of the stages before it.

## Layout

```
data/Al/         the DFT data: training set (7,893 frames), test set, and the 20 % subset MACE was trained on
models/          the fitted models
  Al_12_4_6A_2_/   ACE, 91 parameters:  basis (.json), parameters (lin_params.csv), design matrix, targets,
  Al_16_4_6A_3_/   ACE, 684 parameters  weights and prior (A, Y, W, P .csv) of the fit
  mace/            Al_small_stagetwo.model (MACE as trained) and Al_small_stagetwo_constrained.model (the same
                   network with the constrained readout of Figure 3 written into it)
published/       the results the paper's figures were drawn from, one directory per figure
src/             PhysicsPriors, the Julia library (see the top of src/PhysicsPriors.jl)
python/          mace_readout.py (MACE's readout as a linear model) and neb.py (migration barriers)
scripts/         one script per stage, numbered in the order they run; fit_ace_models.jl and train_mace.sh
                 rebuild the models from the data
run.sh           the SLURM job graph: which stage runs after which, on how many threads
```

## The stages

**Figure 1** (`scripts/fig1_bands/`).  For the 91-parameter model: `al12_1` refits the RLS model with its lattice
constant pinned, Born stability and all phonons above 0.15 THz (the constrained mean); `al12_2` repeats that fit
once per high-leverage observation, each time also reproducing that observation (the constrained POPS cloud, 73,479
fits); `al12_3` rejection-samples the cloud's hypercube into 30 members; `al12_4` draws the 30 unconstrained
members and computes the bands of both ensembles.  For the 684-parameter model the cloud is pinned to the lattice
constant in closed form: `al16_1` samples its hypercube with and without the phonon test, `al16_2` computes the bands.

**Figure 2** (`scripts/fig2_rls_200K/`).  `1` refits the RLS model with phonons constrained at six volumes; `2`
runs 200 K NPT molecular dynamics with either parameter vector and computes its RDF; `3` computes each model's
bands at the lattice constant it reached.

**Figure 3** (`scripts/fig3_mace_elastic/`).  `1a` and `1b` write MACE's readout as a regression problem (two
processes on purpose: see the header of `1b`); `2` refits the 81
readout weights subject to C11, C12, C44 and zero pressure at a0 = 4.032 Å; `3` and `4` build the POPS clouds and
100-member committees of the constrained and unconstrained readout; `5` computes every member's lattice and
elastic constants.

**Figure 4** (`scripts/fig4_migration_barriers/`).  `1` runs climbing-image NEB for the vacancy and the ⟨100⟩
dumbbell with both mean models and all 100 members of both committees.  Every model and every member is relaxed
to its own lattice constant before the defect is built.

## Reproducing the published numbers

The numerical kernels, seeds, solver settings and thread counts are those of the code that produced `published/`
(the paper's repository, `physics-informed-priors-for-mlips-with-uq-arxiv`, v1.0).  What this directory changes is
the organisation: the stages no paper figure reads are gone, and the copies of shared code are one library.

- Sampling and plotting stages are deterministic: `bash run.sh plots` redraws the four published figures pixel
  for pixel.
- The constrained fits stop at a solver tolerance (OSQP, 1e-6 to 1e-9), and their inputs carry round-off that
  depends on the hardware and on the thread counts in `run.sh`.  Run on the same machine, this code and the
  paper's repository return identical parameters; a different machine changes both in the same way.  Two cases
  are sensitive.  The constrained mean of the 91-parameter model has an objective that is flat along some
  directions, so two machines can return parameter vectors that differ in the second decimal while agreeing in
  every constrained quantity.  And a rejection-sampled committee changes member by member as soon as the cloud it
  is drawn from moves in the last digits, because a draw near the edge of the feasible set flips between kept and
  rejected and every later member shifts.  The published results are therefore reproduced exactly only on the
  node type they were computed on.
- `published/fig3/cij_spread.png` is redrawn with the Matplotlib pinned in `requirements.txt` (3.9.4); the copy
  in the paper's repository was drawn with 3.5.1 and differs from it in about 1 % of pixels (text rendering).

## Rebuilding the models

```bash
NPROCS=16 julia --project scripts/fit_ace_models.jl     # → models_refit/Al_12_4_6A_2_/, Al_16_4_6A_3_/
bash scripts/train_mace.sh                              # → models_refit/mace/Al_small_stagetwo.model
ACE_MODELS=models_refit MODEL_FILE=models_refit/mace/Al_small_stagetwo.model bash run.sh all
```

`models/` is never overwritten.

The two ACE design matrices `models/*/A.csv` (247 MB and 1.9 GB) are too large for GitHub and are not in the git
repository.  The stages that need them are the Figure 1 stages and the first Figure 2 stage.  `bash run.sh fit`
rebuilds them from the data into `models_refit/`, from where they can be copied next to the fitted parameters.
