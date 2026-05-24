#!/usr/bin/env bash
set -euo pipefail
cd /home/renku/work/block-factorized-discrete-diffusion

# E2 rerun: offset=-6 sigmoid schedule (floor ~0.0012, low init) + val-ELBO
# checkpoint selection. Fresh output dirs so committed baselines stay intact.
# Tests whether a confound-free (low floor AND low init) learned forward
# schedule learns a graded ramp (FLDD intact) or still saturates (reframe).
#
# 30 epochs: val-ELBO selection peaks at epoch ~21-22 in the quick sweep;
# bs1_s42 100ep gave FID 76.74 vs 30ep 75.28 (identical) -- longer training
# is wasted under val-best selection.

PY=/home/renku/work/.venv/bin/python
export PYTHONUNBUFFERED=1   # stream per-epoch loss/alpha lines live to the log

echo "START $(date -Is)"
"$PY" run_e2.py \
  --T 4 --epochs 30 --batch_size 128 --lr 3e-4 \
  --block_sizes 1 2 4 \
  --seeds 42 43 44 45 46 47 \
  --n_fid_samples 10000 \
  --val_size 5000 \
  --save_dir checkpoints_e2_reparam6 \
  --gen_root fid_stats_e2_reparam6 \
  --results_json results/results_e2_reparam6.json

echo "DONE $(date -Is)"
touch /home/renku/work/block-factorized-discrete-diffusion/.e2_reparam6_done
