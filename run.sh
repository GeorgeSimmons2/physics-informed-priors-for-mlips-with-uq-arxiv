#!/bin/bash
# run.sh — the figures of arXiv:2611.09325.
#
#   bash run.sh setup     the Python environment (.venv/, from requirements.txt) and the Julia one (Manifest.toml)
#   bash run.sh plots     the four figures from the published results → results/              (minutes, no SLURM)
#   bash run.sh all       every stage, from the fitted models in models/ → results/           (SLURM, about a day)
#   bash run.sh fig1 | fig2 | fig3 | fig4        the stages of one figure                     (SLURM)
#   bash run.sh check     compare results/ with published/
#   bash run.sh fit       refit both ACE models from the DFT data → models_refit/           (SLURM, a few hours)
#   bash run.sh fit-here  the same on this machine, in the background (NPROCS workers, default 32)
#
# The job graph below is the recipe the published results were computed with: one 48-core node per job, and the
# thread counts are part of it (the per-basis Hessian sums per-thread buffers, and BLAS follows the allocation,
# so the last bits of a result depend on them).  Env: PARTITION (compute), PY (python; default .venv/bin/python).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
JL="julia --project=$ROOT"
PY=${PY:-$ROOT/.venv/bin/python}
PART=${PARTITION:-compute}
LOG=$ROOT/results/logs
T='-t $SLURM_NTASKS'

# sub NAME DEPENDENCIES TASKS MEM_PER_TASK_MB TIME ENV COMMAND → job id
sub() {
    mkdir -p "$LOG"
    sbatch --parsable -p "$PART" -N 1 -n "$3" --mem-per-cpu="$4" -t "$5" -J "$1" ${2:+-d afterok:$2} \
        -o "$LOG/$1_%j.log" --export="ALL${6:+,$6}" --wrap "cd $ROOT && $7"
}

fig1() {   # phonon bands of unconstrained and constrained POPS ensembles
    local a1 a2 a3 a4 b1 b2 S=scripts/fig1_bands
    a1=$(sub fig1_al12_1 ""        20 3988  06:00:00 "" "$JL $T $S/al12_1_constrained_mean.jl")
    a2=$(sub fig1_al12_2 "$a1"     42 3988  04:00:00 "" "$JL $T $S/al12_2_constrain_cloud.jl")
    a3=$(sub fig1_al12_3 "$a2"      8 12000 04:00:00 "" "$JL $T $S/al12_3_constrained_ensemble.jl")
    a4=$(sub fig1_al12_4 "$a3"     40 3988  08:00:00 "" "$JL $T $S/al12_4_bands.jl")
    b1=$(sub fig1_al16_1 ""        40 3988  06:00:00 "" "$JL $T $S/al16_1_ensembles.jl")
    b2=$(sub fig1_al16_2 "$b1"     40 3988  08:00:00 "" "$JL $T $S/al16_2_bands.jl")
    sub fig1_plot "$a4:$b2"         1 8000  00:30:00 "" "$JL $S/plot.jl"
}

fig2() {   # RLS against constrained RLS at 200 K
    local r n1 n2 b S=scripts/fig2_rls_200K
    r=$(sub  fig2_rls ""           20 7976  12:00:00 "" "$JL $T $S/1_constrained_rls.jl")
    n1=$(sub fig2_npt_rls ""        4 3988  24:00:00 "" "$JL $T $S/2_npt.jl rls")
    n2=$(sub fig2_npt_con "$r"      4 3988  24:00:00 "" "$JL $T $S/2_npt.jl constrained")
    sub fig2_plot "$n1:$n2"         8 3988  02:00:00 "" "$JL $T $S/3_bands_200K.jl && $JL $S/4_plot.jl"
}

fig3() {   # lattice and elastic constants of the constrained MACE readout and its POPS committee
    local E="THREADS=16,OMP_NUM_THREADS=16" m c s S=scripts/fig3_mace_elastic
    m=$(sub fig3_readout ""        16 2000  12:00:00 "$E" "$PY $S/1_readout_rows.py && $JL $S/2_constrained_readout.jl")
    c=$(sub fig3_clouds "$m"       16 2000  06:00:00 "$E" "$JL $T $S/3_pops_clouds.jl")
    s=$(sub fig3_committees "$c"    4 2000  02:00:00 "$E" "$JL $S/4_committees.jl")
    sub fig3_cij "$s"              16 2000  06:00:00 "$E" "$PY $S/5_cij_members.py && $PY $S/6_plot.py"
}

fig4() {   # migration barriers; after fig3 if given its last job id, else from the published committees
    local E="THREADS=16,OMP_NUM_THREADS=16" S=scripts/fig4_migration_barriers
    sub fig4_neb "${1:-}"          16 2000  24:00:00 "$E" "$PY $S/1_neb.py && $PY $S/2_plot.py"
}

case "${1:-help}" in
setup)
    python3 -m venv "$ROOT/.venv"
    "$ROOT/.venv/bin/python" -m pip install --upgrade pip==24.2
    "$ROOT/.venv/bin/python" -m pip install -r requirements.txt
    $JL -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using PhysicsPriors; println("Julia environment ready")'
    ;;
plots)
    $JL scripts/fig1_bands/plot.jl
    $JL scripts/fig2_rls_200K/4_plot.jl
    "$PY" scripts/fig3_mace_elastic/6_plot.py
    "$PY" scripts/fig4_migration_barriers/2_plot.py
    ;;
fig1) fig1 ;;
fig2) fig2 ;;
fig3) fig3 ;;
fig4) fig4 ;;
all)  fig1; fig2; fig4 "$(fig3)" ;;
check) "$PY" scripts/check_against_published.py ;;
fit)   sub fit_ace "" 16 8000 06:00:00 "NPROCS=16" "$JL scripts/fit_ace_models.jl" ;;
fit-here)
    # OPENBLAS_NUM_THREADS=8 is what a 16-task SLURM job gets, so the final solve matches a `fit` job's arithmetic
    mkdir -p "$LOG"
    NPROCS=${NPROCS:-32} OPENBLAS_NUM_THREADS=8 nohup $JL scripts/fit_ace_models.jl > "$LOG/fit_ace_here.log" 2>&1 &
    echo "refit started in the background (pid $!); follow it with: tail -f results/logs/fit_ace_here.log"
    ;;
*)    sed -n 2,15p "$0"; exit 1 ;;
esac
