#!/bin/bash
# Retrain the MACE model from the DFT data (the trained model is already in models/mace/).
#
# 2 interaction layers, 64 channels (64x0e + 64x1o), cutoff 6 Å, correlation order 3, a 16x0e readout MLP; 400
# epochs with EMA and a stochastic-weight-averaging second stage, seed 1, on data/Al/subset_20_percent.extxyz
# (5 % held out for validation).  The figures use the second-stage model, Al_small_stagetwo.model.
#
# Out:  models_refit/mace/Al_small_stagetwo.model     (use it with MODEL_FILE=models_refit/mace/Al_small_stagetwo.model)
# Run:  bash scripts/train_mace.sh                      (about 7 h on 16 cores)
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
DATA=$ROOT/data/Al/subset_20_percent.extxyz
M=$ROOT/models_refit/mace; T=$M/train
mkdir -p "$M" "$T"
cd "$T" && OMP_NUM_THREADS=16 "$PY" -m mace.cli.run_train \
  --name Al_small --train_file "$DATA" --valid_fraction 0.05 --seed 1 \
  --energy_key dft_energy --forces_key dft_forces --E0s average \
  --model MACE --hidden_irreps '64x0e + 64x1o' --r_max 6.0 --num_interactions 2 --max_L 1 \
  --correlation 3 --MLP_irreps 16x0e \
  --batch_size 16 --max_num_epochs 400 --patience 50 --eval_interval 5 --ema --swa \
  --default_dtype float64 --device cpu --save_cpu \
  --model_dir "$M" --checkpoints_dir "$T/checkpoints" --results_dir "$T/results" --log_dir "$T/logs"
