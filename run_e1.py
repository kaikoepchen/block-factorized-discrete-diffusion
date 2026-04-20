"""E1: compare pixel- vs block-factorized reverse models on the synthetic dataset.

Trains the configured block_sizes over multiple seeds and prints a summary
table alongside the exact TV floor an optimal pixel-factorized model would hit.

Per-(block_size, seed) we use the same value for model and data seeds, so
each (|G|=1, seed=s) and (|G|=4, seed=s) pair is trained on identical data.
"""

import argparse
import torch

from train_synthetic import run_synthetic
from fldd.synthetic import pixel_factorized_tv_floor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--block_sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epsilon", type=float, default=0.04)
    parser.add_argument("--n_train", type=int, default=20000)
    parser.add_argument("--n_eval", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="checkpoints_synth")
    parser.add_argument("--samples_dir", type=str, default="samples")
    args = parser.parse_args()

    floor = pixel_factorized_tv_floor(epsilon=args.epsilon)
    print(f"device={args.device} T={args.T} epochs={args.epochs} "
          f"epsilon={args.epsilon}")
    print(f"optimal pixel-factorized TV floor (analytic): {floor:.4f}\n")

    results = []
    for bs in args.block_sizes:
        for seed in args.seeds:
            print(f"--- training |G|={bs} seed={seed} ---")
            r = run_synthetic(
                block_size=bs, seed=seed, data_seed=seed,
                T=args.T, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                n_train=args.n_train, n_eval=args.n_eval,
                epsilon=args.epsilon, device=args.device,
                save_dir=args.save_dir,
                save_samples_path=f"{args.samples_dir}/synth_bs{bs}_s{seed}.png",
                verbose=True,
            )
            print(f"    recon={r['final_recon']:.4f}  "
                  f"block-TV={r['block_tv']:.4f}")
            results.append(r)

    print("\n=== E1 summary ===")
    print(f"analytic pixel-factorized TV floor: {floor:.4f}")
    print(f"{'|G|':>4} | {'recon (mean±std)':>24} | {'block-TV (mean±std)':>24}")
    for bs in args.block_sizes:
        rs = [r for r in results if r["block_size"] == bs]
        recons = torch.tensor([r["final_recon"] for r in rs])
        tvs = torch.tensor([r["block_tv"] for r in rs])
        recon_std = recons.std(unbiased=False) if len(rs) == 1 else recons.std()
        tv_std = tvs.std(unbiased=False) if len(rs) == 1 else tvs.std()
        print(f"{bs:>4} | "
              f"{recons.mean():>10.4f} ± {recon_std:<10.4f} | "
              f"{tvs.mean():>10.4f} ± {tv_std:<10.4f}")


if __name__ == "__main__":
    main()
