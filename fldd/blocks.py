"""Block-factorized output utilities.

Supports block sizes |G| in {1, 2, 4}:
  |G|=1: per-pixel (baseline), Bernoulli per pixel
  |G|=2: 1x2 horizontal blocks, 4-way categorical per block
  |G|=4: 2x2 blocks, 16-way categorical per block
"""

import torch
import torch.nn.functional as F


def pixels_to_blocks(x, block_size):
    """Reshape pixel image into block indices.

    Args:
        x: (B, 1, H, W) binary image, values in {0, 1}
        block_size: 1, 2, or 4

    Returns:
        block_indices: (B, 1, Hb, Wb) integer tensor with values in {0, ..., K^|G|-1}
            where Hb, Wb are the block grid dimensions
    """
    if block_size == 1:
        return x.long()

    B, _, H, W = x.shape

    if block_size == 2:
        # 1x2 horizontal blocks: (B, 1, H, W) -> (B, 1, H, W//2)
        x_blocks = x.reshape(B, 1, H, W // 2, 2)
        # index = p0 * 2 + p1, where p0 is left pixel, p1 is right
        indices = (x_blocks[:, :, :, :, 0] * 2 + x_blocks[:, :, :, :, 1]).long()
        return indices

    if block_size == 4:
        # 2x2 blocks: (B, 1, H, W) -> (B, 1, H//2, W//2)
        x_blocks = x.reshape(B, 1, H // 2, 2, W // 2, 2)
        # index = p00*8 + p01*4 + p10*2 + p11
        p00 = x_blocks[:, :, :, 0, :, 0]
        p01 = x_blocks[:, :, :, 0, :, 1]
        p10 = x_blocks[:, :, :, 1, :, 0]
        p11 = x_blocks[:, :, :, 1, :, 1]
        indices = (p00 * 8 + p01 * 4 + p10 * 2 + p11).long()
        return indices

    raise ValueError(f"unsupported block_size={block_size}")


def block_indices_to_pixels(indices, block_size, H=28, W=28):
    """Convert block indices back to a pixel image.

    Args:
        indices: (B, 1, Hb, Wb) integer block indices
        block_size: 1, 2, or 4
        H, W: output spatial dimensions

    Returns:
        x: (B, 1, H, W) binary image
    """
    if block_size == 1:
        return indices.float()

    B = indices.shape[0]

    if block_size == 2:
        # indices in {0,1,2,3} -> two binary pixels
        p0 = (indices // 2).float()
        p1 = (indices % 2).float()
        # interleave back: (B, 1, H, W//2) -> (B, 1, H, W)
        x = torch.stack([p0, p1], dim=-1)  # (B, 1, H, W//2, 2)
        return x.reshape(B, 1, H, W)

    if block_size == 4:
        # indices in {0,...,15} -> four binary pixels in 2x2
        p00 = ((indices // 8) % 2).float()
        p01 = ((indices // 4) % 2).float()
        p10 = ((indices // 2) % 2).float()
        p11 = (indices % 2).float()
        # reassemble 2x2 blocks
        Hb, Wb = indices.shape[2], indices.shape[3]
        x = torch.zeros(B, 1, H, W, device=indices.device)
        x[:, :, 0::2, 0::2] = p00
        x[:, :, 0::2, 1::2] = p01
        x[:, :, 1::2, 0::2] = p10
        x[:, :, 1::2, 1::2] = p11
        return x

    raise ValueError(f"unsupported block_size={block_size}")


def compute_block_target(pixel_probs, block_size):
    """Compute the target distribution over block states from per-pixel probabilities.

    Since the forward process is element-wise, the target for each block is a
    product of independent Bernoullis. We enumerate all K^|G| joint states.

    Args:
        pixel_probs: (B, 1, H, W) probability that each pixel = 1
        block_size: 1, 2, or 4

    Returns:
        target: (B, K^|G|, Hb, Wb) target distribution over block states
    """
    if block_size == 1:
        # just stack [1-p, p] along channel dim -> (B, 2, H, W)
        return torch.cat([1.0 - pixel_probs, pixel_probs], dim=1)

    B, _, H, W = pixel_probs.shape

    if block_size == 2:
        # 1x2 blocks -> enumerate 4 states: (0,0), (0,1), (1,0), (1,1)
        p = pixel_probs.reshape(B, 1, H, W // 2, 2)
        p0 = p[:, 0, :, :, 0]  # (B, H, W//2) - prob pixel 0 = 1
        p1 = p[:, 0, :, :, 1]  # prob pixel 1 = 1

        # state 0: (0,0) -> (1-p0)(1-p1)
        # state 1: (0,1) -> (1-p0)*p1
        # state 2: (1,0) -> p0*(1-p1)
        # state 3: (1,1) -> p0*p1
        s0 = (1 - p0) * (1 - p1)
        s1 = (1 - p0) * p1
        s2 = p0 * (1 - p1)
        s3 = p0 * p1
        return torch.stack([s0, s1, s2, s3], dim=1)  # (B, 4, H, W//2)

    if block_size == 4:
        # 2x2 blocks -> enumerate 16 states
        p = pixel_probs.reshape(B, 1, H // 2, 2, W // 2, 2)
        p00 = p[:, 0, :, 0, :, 0]  # (B, H//2, W//2)
        p01 = p[:, 0, :, 0, :, 1]
        p10 = p[:, 0, :, 1, :, 0]
        p11 = p[:, 0, :, 1, :, 1]

        # enumerate all 16 states in the same order as pixels_to_blocks
        # state index = b00*8 + b01*4 + b10*2 + b11
        states = []
        for b00 in range(2):
            for b01 in range(2):
                for b10 in range(2):
                    for b11 in range(2):
                        prob = 1.0
                        prob = prob * (p00 if b00 else (1 - p00))
                        prob = prob * (p01 if b01 else (1 - p01))
                        prob = prob * (p10 if b10 else (1 - p10))
                        prob = prob * (p11 if b11 else (1 - p11))
                        states.append(prob)

        return torch.stack(states, dim=1)  # (B, 16, H//2, W//2)

    raise ValueError(f"unsupported block_size={block_size}")


def block_grid_shape(H, W, block_size):
    """Return (Hb, Wb) — spatial dimensions of the block grid."""
    if block_size == 1:
        return H, W
    elif block_size == 2:
        return H, W // 2
    elif block_size == 4:
        return H // 2, W // 2
    raise ValueError(f"unsupported block_size={block_size}")


def num_block_states(block_size, K=2):
    """Number of joint states per block: K^|G|."""
    return K ** block_size
