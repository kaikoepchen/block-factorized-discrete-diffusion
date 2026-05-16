"""Visualize the learned forward-process schedule across E2/E4 checkpoints.

Addresses the "schedule collapse" caveat: report shows αₜ ≈ [0.06, 0.06, 0.06,
0.50] across all 12 E2 runs regardless of block size. This script reads every
best.pt checkpoint in `--ckpt_dirs`, extracts the learned flip-probability
schedule from the LearnedForwardProcess, and produces:

  - viz_schedule_e2.{png,pdf}: per-t αₜ line for every (block_size, seed) on
    E2 checkpoints. Confirms / refutes schedule collapse at T=4.
  - viz_schedule_e4.{png,pdf}: same per E4 T value, one subplot per T,
    color-coded by block size.
  - schedule_summary.json: per-checkpoint αₜ + per-(T, bs) mean/std.
"""

import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

from fldd.forward import LearnedForwardProcess


CKPT_E2_RE = re.compile(r"bs(\d+)_s(\d+)_best\.pt$")
CKPT_E4_RE = re.compile(r"T(\d+)_bs(\d+)_s(\d+)_best\.pt$")
COLORS_BS = {1: "#4393c3", 2: "#7fbf7b", 4: "#d6604d"}


def alphas_from_ckpt(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    T = ckpt["T"]
    fp = LearnedForwardProcess(T=T)
    fp.load_state_dict(ckpt["forward"])
    fp.eval()
    with torch.no_grad():
        alphas = fp.get_alphas().cpu().tolist()
    return T, ckpt["block_size"], int(ckpt["seed"]), alphas


def scan_e2(ckpt_dir):
    out = []
    for path in sorted(glob.glob(os.path.join(ckpt_dir, "bs*_s*_best.pt"))):
        m = CKPT_E2_RE.search(os.path.basename(path))
        if not m:
            continue
        T, bs, seed, alphas = alphas_from_ckpt(path)
        out.append({"T": T, "block_size": bs, "seed": seed, "alphas": alphas,
                    "path": path})
    return out


def scan_e4(ckpt_dir):
    out = []
    for path in sorted(glob.glob(os.path.join(ckpt_dir, "T*_bs*_s*_best.pt"))):
        m = CKPT_E4_RE.search(os.path.basename(path))
        if not m:
            continue
        T, bs, seed, alphas = alphas_from_ckpt(path)
        out.append({"T": T, "block_size": bs, "seed": seed, "alphas": alphas,
                    "path": path})
    return out


def plot_e2(entries, out_prefix):
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    block_sizes = sorted({e["block_size"] for e in entries})
    T = entries[0]["T"]
    xs = np.arange(1, T + 1)

    handles_done = set()
    for e in entries:
        bs = e["block_size"]
        color = COLORS_BS.get(bs, "gray")
        label = f"|G|={bs}" if bs not in handles_done else None
        handles_done.add(bs)
        ax.plot(xs, e["alphas"], marker="o", color=color, alpha=0.55, lw=1.2,
                label=label)

    # overlay per-bs mean
    for bs in block_sizes:
        rows = np.array([e["alphas"] for e in entries if e["block_size"] == bs])
        if len(rows) > 0:
            ax.plot(xs, rows.mean(axis=0), color=COLORS_BS.get(bs, "gray"),
                    lw=2.5, zorder=10)

    ax.set_xticks(xs)
    ax.set_xlabel("diffusion step t")
    ax.set_ylabel(r"learned flip probability  $\alpha_t$")
    ax.set_title(
        "E2: learned forward schedule per run\n"
        "(thin = individual seeds; thick = per-|G| mean)"
    )
    ax.set_ylim(-0.02, 0.55)
    ax.axhline(0.5, ls="--", color="black", lw=0.5, alpha=0.5)
    ax.legend(title="block size", loc="upper left")
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=150)
    fig.savefig(out_prefix + ".pdf")
    plt.close(fig)


def plot_e4(entries, out_prefix):
    if not entries:
        return
    Ts = sorted({e["T"] for e in entries})
    n = len(Ts)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), squeeze=False)
    block_sizes = sorted({e["block_size"] for e in entries})

    for ax, T in zip(axes[0], Ts):
        xs = np.arange(1, T + 1)
        handles_done = set()
        for e in entries:
            if e["T"] != T:
                continue
            bs = e["block_size"]
            color = COLORS_BS.get(bs, "gray")
            label = f"|G|={bs}" if bs not in handles_done else None
            handles_done.add(bs)
            ax.plot(xs, e["alphas"], marker="o", color=color, alpha=0.55,
                    lw=1.0, label=label)
        for bs in block_sizes:
            rows = np.array([e["alphas"] for e in entries
                             if e["T"] == T and e["block_size"] == bs])
            if len(rows) > 0:
                ax.plot(xs, rows.mean(axis=0), color=COLORS_BS.get(bs, "gray"),
                        lw=2.5, zorder=10)
        ax.set_xticks(xs)
        ax.set_xlabel("step t")
        ax.set_title(f"T = {T}")
        ax.set_ylim(-0.02, 0.55)
        ax.axhline(0.5, ls="--", color="black", lw=0.5, alpha=0.5)
        if T == Ts[0]:
            ax.set_ylabel(r"$\alpha_t$")
            ax.legend(title="|G|", loc="upper left", fontsize=8)
    fig.suptitle("E4: learned forward schedule per T", y=1.02)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_prefix + ".pdf", bbox_inches="tight")
    plt.close(fig)


def summarize(entries):
    """Per-(T, bs) mean and std of αₜ across seeds."""
    by_key = {}
    for e in entries:
        key = (e["T"], e["block_size"])
        by_key.setdefault(key, []).append(e["alphas"])
    out = {}
    for (T, bs), rows in by_key.items():
        arr = np.array(rows)
        out[f"T{T}_bs{bs}"] = {
            "T": T, "block_size": bs, "n_seeds": len(rows),
            "alpha_mean": arr.mean(axis=0).tolist(),
            "alpha_std": arr.std(axis=0).tolist(),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2_dir", type=str, default="checkpoints_e2")
    parser.add_argument("--e4_dir", type=str, default="checkpoints_e4")
    parser.add_argument("--out_prefix", type=str, default="viz_schedule")
    parser.add_argument("--summary_json", type=str, default="schedule_summary.json")
    args = parser.parse_args()

    e2_entries = scan_e2(args.e2_dir) if os.path.isdir(args.e2_dir) else []
    e4_entries = scan_e4(args.e4_dir) if os.path.isdir(args.e4_dir) else []

    print(f"E2: scanned {len(e2_entries)} checkpoints in {args.e2_dir}")
    for e in e2_entries:
        a_str = ", ".join(f"{a:.3f}" for a in e["alphas"])
        print(f"  T={e['T']} |G|={e['block_size']} seed={e['seed']}  "
              f"alphas=[{a_str}]")

    print(f"\nE4: scanned {len(e4_entries)} checkpoints in {args.e4_dir}")
    for e in e4_entries:
        a_str = ", ".join(f"{a:.3f}" for a in e["alphas"])
        print(f"  T={e['T']} |G|={e['block_size']} seed={e['seed']}  "
              f"alphas=[{a_str}]")

    plot_e2(e2_entries, args.out_prefix + "_e2")
    plot_e4(e4_entries, args.out_prefix + "_e4")

    payload = {
        "e2": summarize(e2_entries),
        "e4": summarize(e4_entries),
        "per_ckpt_e2": e2_entries,
        "per_ckpt_e4": e4_entries,
    }
    with open(args.summary_json, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: str(o))
    print(f"\nwrote {args.summary_json}")
    print(f"wrote {args.out_prefix}_e2.png/.pdf")
    print(f"wrote {args.out_prefix}_e4.png/.pdf")


if __name__ == "__main__":
    main()
