#!/usr/bin/env bash
set -euo pipefail
cd /home/renku/work/block-factorized-discrete-diffusion

# Quick sanity sweep of offset=-6 reparam + val-ELBO selection.
# 30 epochs x 3 block sizes x 2 seeds = 6 runs (~9 min/run -> ~55 min total).
# Fresh output dirs so the full-100ep artifacts stay separate.

PY=/home/renku/work/.venv/bin/python
export PYTHONUNBUFFERED=1

echo "START $(date -Is)"
"$PY" run_e2.py \
  --T 4 --epochs 30 --batch_size 128 --lr 3e-4 \
  --block_sizes 1 2 4 \
  --seeds 42 43 \
  --n_fid_samples 10000 \
  --val_size 5000 \
  --save_dir checkpoints_e2_reparam6_quick \
  --gen_root fid_stats_e2_reparam6_quick \
  --results_json results/results_e2_reparam6_quick.json

echo "DONE $(date -Is)"
touch /home/renku/work/block-factorized-discrete-diffusion/.e2_reparam6_quick_done
