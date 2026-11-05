#!/bin/bash
# run_neb_pops.sh — migration barriers with POPS uncertainty (mace/6_neb_pops.py) with the cell relaxed for every
# model and every committee member BEFORE the defect is built (CELL=relaxed), all 100 members of both committees.
#
#   bash mace/run_neb_pops.sh            mean bands, all members, then the table and figure   (~10 h on one node)
#   bash mace/run_neb_pops.sh collect    table and figure only, from the member files present
#
# Env: PY (python with mace/ase/torch; default .venv/bin/python), OUT (route root; default repro/saved),
#      N_MEMBERS (100), FMAX (5e-4 eV/Å), THREADS, and 6_neb_pops.py's own knobs (DEFECTS, MEMBERS=a-b for job splitting, ...).
#
# Results written before CELL existed were run with the cell held at A0.  6_neb_pops.py refuses to mix the two
# protocols in one directory, so on first use this script moves those fixed-cell results from
#   $OUT/mace/results/neb_pops/  and  neb_pops.{txt,npz,pdf,png}   to   neb_pops_fixedcell/  and  neb_pops_fixedcell.*
# (nothing is deleted; they stay usable with CELL=fixed).  The relaxed-cell run then writes to neb_pops/ and neb_pops.*,
# which is where 7_cij_vs_barrier.py reads the barriers.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}
export OUT=${OUT:-$REPO/repro/saved}
export N_MEMBERS=${N_MEMBERS:-100}
export CELL=relaxed
export ENDPOINTS=${ENDPOINTS:-member}
export FMAX=${FMAX:-5e-4}          # eV/Å, endpoints and band (6_neb_pops.py's own default is 5e-3)
RES=$OUT/mace/results
STEM=neb_pops; [ "$ENDPOINTS" = mean ] && STEM=neb_pops_meanends
WORK=$RES/$STEM
mkdir -p "$RES"

[ -x "$PY" ] || { echo "no python at $PY: set PY=/path/to/python (needs mace, ase, torch, scipy, matplotlib)"; exit 1; }

# move fixed-cell results (no cell_mode in their .npz, or cell_mode=fixed) out of the way, once
if [ -d "$WORK" ] && "$PY" - "$WORK" <<'PYEOF'
import glob, os, sys
import numpy as np
fs = glob.glob(os.path.join(sys.argv[1], "*.npz"))
stale = [f for f in fs if "cell_mode" not in np.load(f).files or str(np.load(f)["cell_mode"]) != "relaxed"]
sys.exit(0 if stale else 1)        # exit 0 → something to move
PYEOF
then
    OLD=${WORK}_fixedcell
    [ -e "$OLD" ] && { echo "$WORK holds fixed-cell results but $OLD already exists: sort them out by hand"; exit 1; }
    mv "$WORK" "$OLD"
    for ext in txt npz pdf png; do [ -f "$RES/$STEM.$ext" ] && mv "$RES/$STEM.$ext" "$RES/${STEM}_fixedcell.$ext"; done
    echo "fixed-cell results moved: $WORK → $OLD, $RES/$STEM.* → $RES/${STEM}_fixedcell.*"
fi

LOG=$RES/${STEM}_run.log
case "${1:-run}" in
    run)     echo "$(date '+%F %T') CELL=$CELL ENDPOINTS=$ENDPOINTS FMAX=$FMAX N_MEMBERS=$N_MEMBERS MEMBERS=${MEMBERS:-all} → $WORK (log: $LOG)"
             "$PY" "$REPO/mace/6_neb_pops.py" 2>&1 | tee -a "$LOG" ;;
    collect) "$PY" "$REPO/mace/6_neb_pops.py" --collect 2>&1 | tee -a "$LOG" ;;
    *)       echo "usage: bash mace/run_neb_pops.sh [run|collect]"; exit 2 ;;
esac
