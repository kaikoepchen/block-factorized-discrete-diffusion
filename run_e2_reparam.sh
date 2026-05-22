#!/usr/bin/env bash
set -euo pipefail
cd /home/renku/work/block-factorized-discrete-diffusion

# E2 rerun with the non-saturating (CTMC) forward schedule, no structural floor.
# New output dirs so committed baseline checkpoints/results stay intact.
python run_e2.py \
  --T 4 --epochs 100 --batch_size 128 --lr 3e-4 \
  --block_sizes 1 2 4 \
  --seeds 42 43 44 45 46 47 \
  --n_fid_samples 10000 \
  --save_dir checkpoints_e2_reparam \
  --gen_root fid_stats_e2_reparam \
  --results_json results/results_e2_reparam.json

echo "E2_REPARAM_DONE"
touch /home/renku/work/block-factorized-discrete-diffusion/.e2_reparam_done
