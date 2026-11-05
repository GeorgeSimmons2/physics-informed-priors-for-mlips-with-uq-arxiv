# constrained-pops-repro

Reproduces the figures of the paper on physically constrained POPS uncertainty quantification for
aluminium, from a fresh environment, with one script.

## The figures

| figure | made by | output (under `published/` or `repro/<route>/`) |
|---|---|---|
| Phonon bands of unconstrained vs constrained committees, Al_12 (top) and Al_16 (bottom) | `four_panel/plot_four_panel.jl` | `four_panel/bands_four_panel.pdf` |
| RLS vs constrained RLS at 200 K: phonons at a(200 K) and RDFs | `rls_npt/04_fig_200K.jl` | `rls_npt/fig_200K/fig_200K_rls_vs_constrained.pdf` |
| Al_12 test-set parity and calibration of the constrained committee | `al12/03_hypercube_rejection.jl` | `al12/hypercube/{parity,calibration}_fullcloud.pdf` |
| MACE before/after: accuracy and C_ij of the trained model and its constrained readout | `mace/4_compare.py` | `mace/results/compare_elastic.pdf` |
| C_ij of every MACE POPS committee member, constrained vs unconstrained | `mace/pops/p5_figures.py` | `mace/pops/cij_spread.pdf` |

Two follow-up figures on the MACE committees, beyond the paper (in `published/mace/results/` from the 100-member run):

| figure | made by | output (under `repro/<route>/`) |
|---|---|---|
| Vacancy and ⟨100⟩-dumbbell migration barriers of both mean models and of every committee member | `mace/6_neb_pops.py` | `mace/results/neb_pops.pdf` |
| Every MACE committee member's a0, C11, C12, C44 against its vacancy migration barrier: is the barrier spread elastic? | `mace/7_cij_vs_barrier.py` | `mace/results/cij_vs_barrier_vacancy.pdf` |

What is computed:
- **Al_12_4_6A_2_ (91 parameters).** Each member of the POPS cloud (the top-50%-leverage observations,
  73,479 members) is refitted with the lattice constant pinned, Born stability, and cutting planes until
  every phonon on the band path is above 0.15 THz at 0 K (`al12/01`, `al12/02`). A hypercube around the
  constrained cloud is then rejection-sampled on the same check (`al12/03`). `al12/04` draws the
  unconstrained ensemble and relaxes and computes the bands of every member; `four_panel/build_bands_cache_Al_12.jl`
  recomputes both ensembles' band curves (at the recorded lattice constants) into the cache the figure reads.
- **Al_16_4_6A_3_ (684 parameters).** The cloud is pinned to the lattice constant in closed form and the
  hypercube is rejection-sampled on phonon stability (`al16/01`); `al16/02` relaxes every member and computes its bands.
- **RLS vs constrained RLS.** Al_12 is refitted with the lattice constant, Born stability and phonons
  above 0.01 THz at every volume from 1.00 to 1.10 a_eq (`rls_npt/01`). Both models run 200 K NPT with the
  same seed (`02`); the RLS model leaves FCC, the constrained one does not (`03`, `04`).
- **MACE.** The readout of the 64-channel MACE model is refitted under a0 = 4.032 Å and
  C11/C12/C44 = 116.3/64.8/30.9 GPa (`mace/1`–`3`) and compared with the trained model (`4`). Constrained and
  unconstrained POPS committees of the readout are built and evaluated in `mace/pops/p1`–`p4`. `mace/6` then runs
  climbing-image NEB migration barriers (vacancy; ⟨100⟩ split-dumbbell interstitial) for both mean models and, from
  each mean band, for the first 30 members of each committee; `mace/7` plots every member's elastic constants
  against its barrier.

## Running it

```bash
bash run.sh setup     # .venv/ and a repo-local Julia depot .julia/                       ~40 min, once
bash run.sh data      # the tier-B inputs from ../physics-informed-priors-repro, SHA-256 checked
bash run.sh replot    # every figure from published/ → repro/replot/                      minutes
bash run.sh saved     # recompute everything from the published fitted models → repro/saved/   SLURM
bash run.sh test      # unit tests (src/, lib/, every function the scripts define) + replot regression  ~25 min
```

Needs `julia` 1.11, `python3` 3.9 and SLURM (`PARTITION`, default `compute`). Every job fits on one
48-core node with about 170 GB of memory.

`published/` holds the last full rebuild from scratch (15–16 Oct 2026; constrained RLS 25 Oct). Reruns
reproduce its figures within numerical tolerance.

## Pipeline

```
Al_12   al12/01 ─► al12/02 ─► al12/03 ─► al12/04 ─► build_bands_cache_Al_12 ─┐
Al_16   al16/01 ─► al16/02 ───────────────────────────────────────────────────┴─► plot_four_panel
RLS     rls_npt/01 ─► 02 (constrained) ─┐
                      02 (RLS) ─────────┴─► 03 + 04 (fig_200K)
MACE    1 → 2 → 3_qp ─► 4 → 5
        p1 → p1b ─(+3_qp)─► p2 → p3 → p4 → p5 ─► 6 (NEB) ─► 7        (6 and 7 are not in run.sh: see Extras)
```

## Layout

```
run.sh                       setup | data | replot | saved   (extras: refit | all | figures)
src/                         ACEWorkflow: model build/load, POPS samplers, elastic Hessian, phonon band path
lib/bandpath.jl              per-basis Hessian (cached beside each model), band path, stability helpers
lib/parity_calibration.jl    test-set parity and calibration figures
al12/01_committee.jl         constrained mean model and the 30-member repaired forest
al12/02_cutting_plane_cloud.jl  constrain the whole cloud with cutting planes
al12/03_hypercube_rejection.jl  hypercube around the cloud, rejection sampling, parity/calibration
al12/03_replot_parity.jl     redraw al12/03's figures from its saved predictions
al12/04_naive_ensemble.jl    the unconstrained ensemble
al16/01_pinned_ensembles.jl  pinned hypercube: unconstrained and rejection-sampled ensembles
al16/02_bands.jl             relaxation and bands of every Al_16 member
four_panel/                  Al_12 band cache, then the 2×2 figure; check_rejection_rate.jl re-measures the samplers' acceptance
rls_npt/                     01 constrained RLS, 02 NPT, 03 RDF, 04 the figure
mace/                        1–3 constrained readout, 4 before/after, 5 matscipy check, 6 POPS migration barriers, 7 C_ij vs
                             barrier; pops/ p1–p5 committees.  vacancy_migration_barrier.py and neb_interstitial.py are the
                             earlier two-model NEB scripts (superseded by 6); their runs are in mace/results/
ace_models/, mace/train.sh   refit both ACE models / retrain MACE from the DFT data (refit route)
models/, mace/models/        the published fitted models (mace/models/ also holds Al_small_stagetwo_elastic.model, see Extras)
published/                   the published results, read by replot
```

Scripts are copies of the ones that produced the paper's results, with paths redirected to this repository's
layout; each header names its source, and remaining changes are marked `[PORT]`.
Paper figures are drawn at their printed width (510 pt full, 255 pt half) with 11–13 pt text.

## Extras

- `bash run.sh refit` refits both ACE models and retrains MACE (~5.5 h) from the DFT data, then runs the
  same chain into `repro/refit/`. `bash run.sh all` runs both routes.
- `bash run.sh figures ROUTE` redraws a finished route's figures from its results.
- `mace/5_elastic_matscipy.py` checks the MACE C_ij independently with matscipy.
- `mace/6_neb_pops.py` (not in `run.sh`): migration barriers with POPS uncertainty — 3³ cubic cell, 7-image
  climbing-image NEB, FMAX 5e-3 eV/Å. Every model and every committee member sits at its **own relaxed lattice
  constant** (`CELL=relaxed`, the default): the perfect fcc cell is relaxed for the readout in use *before* the defect
  is built, and the endpoints and band are then relaxed internally at that cell. The two mean models' bands first
  (IDPP start), then the first `N_MEMBERS` (30) members of each committee — the members of an N = 30 `p3_sample.jl`
  run — each relax their own cell, scale the mean band to it, and re-relax both endpoints and the band from there, so
  a member costs a few tens of NEB steps. Every member is checked to have relaxed to the mean model's transition
  (`path` in the table: no atom more than `PATH_TOL` = 0.5 Å from the scaled mean band, same saddle image, symmetric
  endpoints, converged). `ENDPOINTS=mean` keeps the mean model's endpoint geometries (scaled to the member's cell)
  and writes to `neb_pops_meanends.*` instead; `CELL=fixed` holds every cell at A0 = 4.032 Å (the old protocol) and
  appends `_fixedcell` to the output names. Reads θc and the committees from `$OUT/mace/` (falls back to
  `published/mace/`). Members are independent: `MEMBERS=0-14 .venv/bin/python mace/6_neb_pops.py` per job, then
  `--collect` once every member file exists. `bash mace/run_neb_pops.sh` runs all 100 members of both committees
  with relaxed cells (moving any fixed-cell results already in `neb_pops/` to `neb_pops_fixedcell*` first);
  `bash mace/run_neb_pops.sh collect` redraws the table and figure.
  Out: `$OUT/mace/results/neb_pops.{txt,npz,pdf,png}` and the per-member files (with each member's a0) in `neb_pops/`.
- `mace/7_cij_vs_barrier.py`: every member's a0, C11, C12, C44 (`cij_members.csv`, p4) against its barrier
  (`neb_pops.npz`, 6), member by member, with Pearson and Spearman correlations and a least-squares slope. The table
  covers both committees; the figure draws the MACE (unconstrained) committee only, since the constrained members'
  C_ij are pinned within ±0.1 % (`COMMITTEES=unconstrained,constrained` draws both rows); `DEFECTS=vacancy,interstitial`
  draws both defects. Out: `$OUT/mace/results/cij_vs_barrier_<defect>.{pdf,png,txt}`.
- `four_panel/check_rejection_rate.jl`: re-measures the acceptance rate of the constrained hypercube rejection sampler
  for Al_12 and Al_16 from each study's own cloud and predicate (`$OUT/four_panel/rejection_rate.txt`; the Al_12 cloud
  needs a `saved` route).
- `mace/models/Al_small_stagetwo_elastic.model`, used by the two earlier NEB scripts, is the constrained readout of
  `../physics-informed-priors-repro/mace_al_elastic` (byte-identical): the same C11/C12/C44 targets and ±0.1 % windows,
  but imposed at A0 = 4.04 Å, the lattice constant that study used before this one moved to 4.032 Å. Its own a0 is
  therefore 4.040 Å, and by the 4.032 Å constraint rows of `mace/1` it reads C11/C12/C44 = 119.6/65.9/32.3 GPa with
  P(4.032 Å) = −0.5 GPa, against 116.4/64.7/30.9 and 0 for the published θc
  (`published/mace/results/Al_small_stagetwo_elastic.model`, constrained at 4.032 Å). Everything `run.sh` runs, and
  `mace/6`–`7`, use θc.
- Further outputs: MACE POPS parity plots and test-set numbers (`mace/pops/parity_*`, `uq_test.txt`),
  the constrained RLS bands at each volume (`rls_npt/constrained/`), the Al_12 cloud diagnostics
  (`al12/cloud/member_diagnostics.csv`).

## Data

- **In git:** code, the published fitted models, `published/`, and the two earlier NEB scripts' runs (`mace/results/`).
- **Not in git** (`data/tierB.sha256`): the DFT training and test sets and the two ACE design matrices
  `models/*/A.csv`. `bash run.sh data` hardlinks or copies them in and checks every SHA-256.
- **Not in git either:** `repro/` — every route's output, including the POPS NEB results that `mace/6` and `mace/7`
  write. Copy `neb_pops.{txt,npz,pdf,png}` and `cij_vs_barrier_*` into `published/mace/results/` to publish them.
