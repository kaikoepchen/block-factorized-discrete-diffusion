#!/usr/bin/env bash
set -euo pipefail
cd /home/renku/work/block-factorized-discrete-diffusion

echo "==== [1/5] E2 1v4 (regenerate merged) ===="
python merge_e2_stats.py --bs_a 1 --bs_b 4 \
  --out results/results_e2_merged.json \
  --sources results/results_e2_from_ckpts.json \
            results/results_e2_extra_bs1_s45.json \
            results/results_e2_extra_bs1.json \
            results/results_e2_extra_bs4.json

echo "==== [2/5] E2 1v2 ===="
python merge_e2_stats.py --bs_a 1 --bs_b 2 \
  --out results/results_e2_merged_1v2.json \
  --sources results/results_e2_merged.json results/results_e2_bs2.json

echo "==== [3/5] E2 2v4 ===="
python merge_e2_stats.py --bs_a 2 --bs_b 4 \
  --out results/results_e2_merged_2v4.json \
  --sources results/results_e2_merged.json results/results_e2_bs2.json

echo "==== [4/5] Holm correction ===="
python holm_e2.py

echo "==== [5/5] E4 paired (no bootstrap CI) + corrected ELBO ===="
python merge_e4_stats.py
python recompute_elbo.py

echo "ALL_DONE"
touch /home/renku/work/block-factorized-discrete-diffusion/.fixes_done
