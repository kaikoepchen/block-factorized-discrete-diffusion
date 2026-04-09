import torch
import os
from torchvision.utils import save_image


@torch.no_grad()
def sample(model, forward_process, T, n_samples=64, device="cpu"):
    """Generate samples by running the reverse process.

    Start from z_T ~ Uniform({0,1}) and iteratively apply
    p_theta(z_{t-1} | z_t) for t = T, T-1, ..., 1.

    Returns:
        x: (n_samples, 1, 28, 28) generated binary images
    """
    model.eval()
    forward_process.eval()

    # start from uniform noise
    z = torch.bernoulli(0.5 * torch.ones(n_samples, 1, 28, 28, device=device))

    for t in range(T, 0, -1):
        t_batch = torch.full((n_samples,), t - 1, device=device, dtype=torch.long)
        logits = model(z, t_batch)
        probs = torch.sigmoid(logits)

        if t > 1:
            # sample z_{t-1}
            z = torch.bernoulli(probs)
        else:
            # final step: threshold at 0.5 for deterministic output
            z = (probs > 0.5).float()

    return z


def save_samples(samples, path, nrow=8):
    """Save a grid of samples as an image."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(samples, path, nrow=nrow)
