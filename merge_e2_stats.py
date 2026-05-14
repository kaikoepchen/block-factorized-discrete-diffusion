"""Merge E2 seeds across runs and run paired statistics on FID + ELBO.

Reads results_e2_from_ckpts.json (existing seeds, FID re-scored from best.pt)
and results_e2_extra.json (new seeds from run_e2.py), pairs runs by seed
across |G|=1 / |G|=4, and reports paired t-test, Wilcoxon signed-rank,
sign test, and a bootstrap CI on the mean paired difference.
"""

import argparse
import json
import math
from collections import defaultdict

import numpy as np


def load_runs(path, fid_field="fid", loss_field=None):
    """Return list of dicts: [{block_size, seed, fid, loss?}, ...]."""
    with open(path) as f:
        d = json.load(f)
    out = []
    for r in d["per_run"]:
        entry = {"block_size": int(r["block_size"]), "seed": int(r["seed"]),
                 "fid": float(r[fid_field])}
        if loss_field and loss_field in r:
            entry["loss"] = float(r[loss_field])
        out.append(entry)
    return out


def collect_pairs(runs, bs_a=1, bs_b=4, metric="fid"):
    """Per-seed paired (a, b) values where a=|G|=bs_a, b=|G|=bs_b."""
    by_seed = defaultdict(dict)
    for r in runs:
        if metric not in r:
            continue
        by_seed[r["seed"]][r["block_size"]] = r[metric]
    pairs = []
    for seed in sorted(by_seed):
        sd = by_seed[seed]
        if bs_a in sd and bs_b in sd:
            pairs.append((seed, sd[bs_a], sd[bs_b]))
    return pairs


def paired_t(diffs):
    n = len(diffs)
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = mean / se if se > 0 else float("inf")
    # two-sided p-value via Student's t; use scipy if available, else
    # rough normal approximation (n is small so we want exact t).
    try:
        from scipy.stats import t as student_t
        p_two = 2.0 * (1.0 - student_t.cdf(abs(t), df=n - 1))
        p_one = 1.0 - student_t.cdf(t, df=n - 1)  # H1: mean > 0
    except ImportError:
        from math import erf, sqrt
        # normal approx (conservative bias for small n)
        p_two = 2.0 * (1.0 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
        p_one = 1.0 - 0.5 * (1 + erf(t / sqrt(2)))
    return {"n": n, "mean_diff": mean, "sd_diff": sd, "t": t,
            "p_two_sided": p_two, "p_one_sided": p_one}


def wilcoxon(diffs):
    try:
        from scipy.stats import wilcoxon as _w
        res = _w(diffs, alternative="greater")
        return {"statistic": float(res.statistic), "p_one_sided": float(res.pvalue)}
    except Exception:
        return None


def sign_test(diffs):
    n = len(diffs)
    pos = sum(d > 0 for d in diffs)
    # one-sided p: P(X >= pos | n trials, p=0.5)
    from math import comb
    p = sum(comb(n, k) for k in range(pos, n + 1)) / (2 ** n)
    return {"n": n, "n_positive": pos, "p_one_sided": p}


def bootstrap_ci(diffs, n_boot=20000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    n = len(diffs)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boots[i] = sample.mean()
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return {"mean": float(diffs.mean()), "ci": ci, "lo": lo, "hi": hi,
            "n_boot": n_boot}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", type=str,
                        default="results_e2_from_ckpts.json")
    parser.add_argument("--extra", type=str, default="results_e2_extra.json")
    parser.add_argument("--out", type=str, default="results_e2_merged.json")
    args = parser.parse_args()

    runs_a = load_runs(args.existing, fid_field="fid", loss_field="ckpt_loss")
    runs_b = load_runs(args.extra, fid_field="fid", loss_field="best_loss")

    # tag source for traceability
    for r in runs_a:
        r["source"] = "from_ckpts"
    for r in runs_b:
        r["source"] = "extra"

    all_runs = runs_a + runs_b
    print(f"loaded {len(runs_a)} from {args.existing}, {len(runs_b)} from {args.extra}")
    print(f"  total {len(all_runs)} runs\n")

    fid_pairs = collect_pairs(all_runs, 1, 4, "fid")
    loss_pairs = collect_pairs(all_runs, 1, 4, "loss")

    print("=== per-seed table (|G|=1 vs |G|=4) ===")
    print(f"{'seed':>5} | {'fid bs1':>9}  {'fid bs4':>9}  {'fid Δ':>8}  |  "
          f"{'loss bs1':>9}  {'loss bs4':>9}  {'loss Δ':>8}")
    by_seed = {s: {"fid": (a, b)} for s, a, b in fid_pairs}
    for s, a, b in loss_pairs:
        by_seed.setdefault(s, {})["loss"] = (a, b)
    for s in sorted(by_seed):
        e = by_seed[s]
        f_str = (f"{e['fid'][0]:>9.4f}  {e['fid'][1]:>9.4f}  "
                 f"{e['fid'][0]-e['fid'][1]:>+8.4f}") if "fid" in e else " " * 32
        l_str = (f"{e['loss'][0]:>9.4f}  {e['loss'][1]:>9.4f}  "
                 f"{e['loss'][0]-e['loss'][1]:>+8.4f}") if "loss" in e else ""
        print(f"{s:>5} | {f_str}  |  {l_str}")

    fid_diffs = [a - b for _, a, b in fid_pairs]    # positive => |G|=4 better
    loss_diffs = [a - b for _, a, b in loss_pairs]  # positive => |G|=4 better

    print(f"\n=== FID (n={len(fid_diffs)} paired) ===")
    print(f"  bs1 mean = {np.mean([a for _, a, _ in fid_pairs]):.4f}")
    print(f"  bs4 mean = {np.mean([b for _, _, b in fid_pairs]):.4f}")
    ttest = paired_t(fid_diffs)
    print(f"  paired t-test:  t={ttest['t']:.3f}  p(one-sided)={ttest['p_one_sided']:.4f}"
          f"  p(two-sided)={ttest['p_two_sided']:.4f}")
    print(f"  mean Δ = {ttest['mean_diff']:.4f}  (sd Δ = {ttest['sd_diff']:.4f})")
    ci = bootstrap_ci(fid_diffs)
    print(f"  bootstrap 95% CI on Δ: [{ci['lo']:.4f}, {ci['hi']:.4f}]")
    w = wilcoxon(fid_diffs)
    if w:
        print(f"  Wilcoxon signed-rank: stat={w['statistic']:.3f}  "
              f"p(one-sided)={w['p_one_sided']:.4f}")
    st = sign_test(fid_diffs)
    print(f"  sign test: {st['n_positive']}/{st['n']} favor |G|=4, "
          f"p(one-sided)={st['p_one_sided']:.4f}")

    print(f"\n=== ELBO loss (n={len(loss_diffs)} paired) ===")
    ttest_l = paired_t(loss_diffs)
    print(f"  paired t-test:  t={ttest_l['t']:.3f}  "
          f"p(one-sided)={ttest_l['p_one_sided']:.4f}")
    print(f"  mean Δ = {ttest_l['mean_diff']:.4f}  (sd Δ = {ttest_l['sd_diff']:.4f})")
    ci_l = bootstrap_ci(loss_diffs)
    print(f"  bootstrap 95% CI on Δ: [{ci_l['lo']:.4f}, {ci_l['hi']:.4f}]")

    payload = {
        "fid": {
            "pairs": [{"seed": s, "bs1": a, "bs4": b, "diff": a - b}
                      for s, a, b in fid_pairs],
            "t_test": ttest,
            "wilcoxon": w,
            "sign_test": st,
            "bootstrap_ci": ci,
        },
        "loss": {
            "pairs": [{"seed": s, "bs1": a, "bs4": b, "diff": a - b}
                      for s, a, b in loss_pairs],
            "t_test": ttest_l,
            "bootstrap_ci": ci_l,
        },
        "sources": {"existing": args.existing, "extra": args.extra},
        "per_run": all_runs,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote merged results -> {args.out}")


if __name__ == "__main__":
    main()
