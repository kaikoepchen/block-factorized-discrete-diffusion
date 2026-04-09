import argparse
import os
import torch
from tqdm import tqdm

from fldd.data import get_binarized_mnist
from fldd.forward import LearnedForwardProcess
from fldd.unet import UNet
from fldd.train import train_epoch
from fldd.sample import sample, save_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4, help="number of diffusion steps")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--sample_every", type=int, default=10, help="generate samples every N epochs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    print(f"Training FLDD on binarized MNIST, T={args.T}, device={args.device}")

    # data
    train_loader, test_loader = get_binarized_mnist(batch_size=args.batch_size)

    # models
    forward_process = LearnedForwardProcess(T=args.T).to(args.device)
    model = UNet(channels=(32, 64, 128)).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(forward_process.parameters()),
        lr=args.lr,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    best_loss = float("inf")

    for epoch in tqdm(range(1, args.epochs + 1), desc="training"):
        metrics = train_epoch(model, forward_process, train_loader, optimizer, args.T, args.device)

        alphas = forward_process.get_alphas().detach().cpu().tolist()
        alpha_str = ", ".join(f"{a:.4f}" for a in alphas)

        print(f"epoch {epoch:3d} | loss {metrics['loss']:.4f} | "
              f"recon {metrics['recon']:.4f} | prior {metrics['prior']:.4f} | "
              f"alphas [{alpha_str}]")

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "forward": forward_process.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": best_loss,
            }, os.path.join(args.save_dir, "best.pt"))

        if epoch % args.sample_every == 0:
            samples = sample(model, forward_process, args.T, n_samples=64, device=args.device)
            save_samples(samples, f"samples/epoch_{epoch:03d}.png")
            print(f"  -> saved samples to samples/epoch_{epoch:03d}.png")

    # save final checkpoint
    torch.save({
        "epoch": args.epochs,
        "model": model.state_dict(),
        "forward": forward_process.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": metrics["loss"],
    }, os.path.join(args.save_dir, "final.pt"))

    # generate final samples
    samples = sample(model, forward_process, args.T, n_samples=64, device=args.device)
    save_samples(samples, "samples/final.png")
    print("done. final samples saved to samples/final.png")


if __name__ == "__main__":
    main()
