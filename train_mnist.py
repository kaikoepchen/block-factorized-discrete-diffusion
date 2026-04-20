import argparse
import os
import torch
from tqdm import tqdm

from fldd.data import get_binarized_mnist
from fldd.forward import LearnedForwardProcess
from fldd.unet import UNet
from fldd.train import train_epoch
from fldd.sample import sample, save_samples


def run_mnist(block_size=1, seed=42, T=4, epochs=100, batch_size=128, lr=3e-4,
              device="cuda", save_dir="checkpoints",
              save_ckpt_as_best="best.pt", save_ckpt_as_final="final.pt",
              sample_every=10, samples_dir="samples", verbose=True):
    """Train FLDD on binarized MNIST.

    Returns a dict with the trained model, forward process, final/best loss,
    and the learned alpha schedule. Pass `save_ckpt_as_best=None` or
    `save_ckpt_as_final=None` to skip those writes; set `sample_every=0` to
    disable intermediate sample grids.
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    if verbose:
        print(f"Training FLDD on binarized MNIST, T={T}, "
              f"block_size={block_size}, seed={seed}, device={device}")

    train_loader, _ = get_binarized_mnist(batch_size=batch_size)

    forward_process = LearnedForwardProcess(T=T).to(device)
    model = UNet(channels=(32, 64, 128), block_size=block_size).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"U-Net parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(forward_process.parameters()),
        lr=lr,
    )

    os.makedirs(save_dir, exist_ok=True)
    if sample_every and samples_dir:
        os.makedirs(samples_dir, exist_ok=True)

    best_loss = float("inf")
    best_epoch = None
    metrics = None

    for epoch in tqdm(range(1, epochs + 1), desc="training", disable=not verbose):
        metrics = train_epoch(model, forward_process, train_loader, optimizer,
                              T, device, block_size)

        alphas = forward_process.get_alphas().detach().cpu().tolist()
        alpha_str = ", ".join(f"{a:.4f}" for a in alphas)
        if verbose:
            print(f"epoch {epoch:3d} | loss {metrics['loss']:.4f} | "
                  f"recon {metrics['recon']:.4f} | prior {metrics['prior']:.4f} | "
                  f"alphas [{alpha_str}]")

        if metrics["loss"] < best_loss and save_ckpt_as_best is not None:
            best_loss = metrics["loss"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "forward": forward_process.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": best_loss,
                "block_size": block_size,
                "T": T,
                "seed": seed,
            }, os.path.join(save_dir, save_ckpt_as_best))

        if (sample_every and sample_every > 0 and samples_dir
                and epoch % sample_every == 0):
            samples = sample(model, forward_process, T, n_samples=64,
                             device=device, block_size=block_size)
            save_samples(samples, os.path.join(samples_dir, f"epoch_{epoch:03d}.png"))
            if verbose:
                print(f"  -> saved samples to {samples_dir}/epoch_{epoch:03d}.png")

    if save_ckpt_as_final is not None:
        torch.save({
            "epoch": epochs,
            "model": model.state_dict(),
            "forward": forward_process.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": metrics["loss"],
            "block_size": block_size,
            "T": T,
            "seed": seed,
        }, os.path.join(save_dir, save_ckpt_as_final))

    return {
        "model": model,
        "forward_process": forward_process,
        "block_size": block_size,
        "seed": seed,
        "T": T,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "final_loss": metrics["loss"],
        "final_recon": metrics["recon"],
        "final_alphas": forward_process.get_alphas().detach().cpu().tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4, help="number of diffusion steps")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--block_size", type=int, default=1, choices=[1, 2, 4],
                        help="block factorization size: 1=pixel, 2=1x2, 4=2x2")
    parser.add_argument("--sample_every", type=int, default=10,
                        help="generate samples every N epochs (0 disables)")
    args = parser.parse_args()

    result = run_mnist(
        block_size=args.block_size, seed=args.seed,
        T=args.T, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, save_dir=args.save_dir,
        sample_every=args.sample_every, samples_dir="samples",
        verbose=True,
    )

    samples = sample(result["model"], result["forward_process"], args.T,
                     n_samples=64, device=args.device,
                     block_size=args.block_size)
    save_samples(samples, "samples/final.png")
    print("done. final samples saved to samples/final.png")


if __name__ == "__main__":
    main()
