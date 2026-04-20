"""E1 synthetic dataset: 28x28 binary images tiled with i.i.d. 2x2 blocks.

Each block is drawn from a ground-truth categorical over 16 joint states,
heavily peaked on 4 strongly-within-block-correlated patterns (all-0, all-1,
and two 2x2 checkers) with an epsilon-uniform noise floor so every state has
strictly positive support (avoids log(0)).

By symmetry of the peak set, every pixel marginal is exactly 0.5. A perfectly
trained pixel-factorized model can therefore only represent the uniform
distribution over 16 states, giving a known irreducible TV floor against the
ground truth. A block-factorized (|G|=4) model can in principle match the
joint exactly, so the block-TV metric cleanly separates the two.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from fldd.blocks import pixels_to_blocks, block_indices_to_pixels


# state index convention matches blocks.py: s = p00*8 + p01*4 + p10*2 + p11
# 0  -> (0,0,0,0) all-zero
# 6  -> (0,1,1,0) checker A
# 9  -> (1,0,0,1) checker B
# 15 -> (1,1,1,1) all-one
PEAK_STATES = (0, 6, 9, 15)


def get_ground_truth_block_dist(epsilon=0.04, peak_states=PEAK_STATES, n_states=16):
    """Option (c): peaked on correlated 2x2 patterns plus uniform noise.

    Mass (1 - epsilon) split equally over peak_states; epsilon spread
    uniformly across all n_states.
    """
    p = torch.full((n_states,), epsilon / n_states)
    peak_mass = (1.0 - epsilon) / len(peak_states)
    for s in peak_states:
        p[s] = p[s] + peak_mass
    return p


def sample_synthetic_images(n_images, H=28, W=28, dist=None, generator=None):
    """Generate binary images by tiling with i.i.d. 2x2 block samples from `dist`."""
    assert H % 2 == 0 and W % 2 == 0
    if dist is None:
        dist = get_ground_truth_block_dist()

    Hb, Wb = H // 2, W // 2
    n_blocks = n_images * Hb * Wb

    state_idx = torch.multinomial(
        dist, n_blocks, replacement=True, generator=generator
    )
    indices = state_idx.reshape(n_images, 1, Hb, Wb)
    return block_indices_to_pixels(indices, block_size=4, H=H, W=W)


def get_synthetic_dataset(n_train=20000, n_test=5000, batch_size=128,
                          seed=0, epsilon=0.04):
    """Build train/test loaders over the synthetic block-tiled dataset."""
    g_train = torch.Generator().manual_seed(seed)
    g_test = torch.Generator().manual_seed(seed + 10_000)
    dist = get_ground_truth_block_dist(epsilon=epsilon)

    train_imgs = sample_synthetic_images(n_train, generator=g_train, dist=dist)
    test_imgs = sample_synthetic_images(n_test, generator=g_test, dist=dist)

    train_loader = DataLoader(
        TensorDataset(train_imgs),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_imgs),
        batch_size=batch_size, shuffle=False,
    )
    return train_loader, test_loader, dist


def empirical_block_dist(imgs, n_states=16):
    """Empirical histogram of 2x2 block states, averaged over all positions."""
    idx = pixels_to_blocks(imgs, block_size=4).flatten()
    hist = torch.bincount(idx, minlength=n_states).float()
    return hist / hist.sum()


def tv_distance(p, q):
    return 0.5 * (p - q).abs().sum().item()


def pixel_factorized_tv_floor(epsilon=0.04):
    """Exact TV achievable by an optimal pixel-factorized model.

    By symmetry of the peak set, pixel marginals are 0.5, so the best
    per-pixel Bernoulli model induces Uniform(16) over block states.
    """
    true = get_ground_truth_block_dist(epsilon=epsilon)
    uniform = torch.full((16,), 1.0 / 16.0)
    return tv_distance(uniform, true)
