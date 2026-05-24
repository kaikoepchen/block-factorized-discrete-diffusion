#!/usr/bin/env bash
set -euo pipefail
cd /home/renku/work/block-factorized-discrete-diffusion

# One-shot rerun of E1 + E3 + E4 at OFFSET=6 + val-ELBO selection so that
# every committed result lives in a single coherent parameterization regime.
# E2 was already done at reparam6 (results/results_e2_reparam6.json).
# 30 epochs everywhere; this is the stated time-budget caveat.

PY=/home/renku/work/.venv/bin/python
export PYTHONUNBUFFERED=1

mkdir -p checkpoints_e3_reparam6 checkpoints_e4_t4_reparam6

# symlink reparam6 valbest checkpoints as _best.pt so run_e3 / run_e4's regex
# (which looks for bs{X}_s{Y}_best.pt) picks up the val-ELBO selected models.
for seed in 42 43 44 45 46 47; do
  ln -sf "../checkpoints_e2_reparam6/bs4_s${seed}_valbest.pt" \
         "checkpoints_e3_reparam6/bs4_s${seed}_best.pt"
done
for seed in 42 43 44; do
  for bs in 1 4; do
    ln -sf "../checkpoints_e2_reparam6/bs${bs}_s${seed}_valbest.pt" \
           "checkpoints_e4_t4_reparam6/bs${bs}_s${seed}_best.pt"
  done
done

echo "=== E1 (synthetic, 30ep, |G|=1,4, seeds 42-44) ==="
echo "START $(date -Is)"
"$PY" run_e1.py --device cuda --epochs 30 \
  --seeds 42 43 44 --block_sizes 1 4 \
  --results_json results/results_e1_reparam6.json

echo
echo "=== E3 (within-block TC on reparam6 |G|=4 valbest, 6 seeds) ==="
echo "START $(date -Is)"
"$PY" run_e3.py --device cuda \
  --ckpt_dir checkpoints_e3_reparam6 \
  --results_json results/results_e3_reparam6.json \
  --fig_prefix figures/e3_reparam6

echo
echo "=== E4 (T sweep, offset=6, 30ep, T={2,4,8,16} x |G|={1,4} x seeds 42-44) ==="
echo "START $(date -Is)"
"$PY" run_e4.py --device cuda \
  --T_values 2 4 8 16 --block_sizes 1 4 --seeds 42 43 44 \
  --epochs 30 --n_fid_samples 10000 \
  --save_dir checkpoints_e4_reparam6 \
  --reuse_t4_dir checkpoints_e4_t4_reparam6 \
  --gen_root fid_stats_e4_reparam6 \
  --results_json results/results_e4_reparam6.json \
  --fig_prefix figures/e4_reparam6

echo
echo "DONE $(date -Is)"
touch /home/renku/work/block-factorized-discrete-diffusion/.clean_reparam6_done
