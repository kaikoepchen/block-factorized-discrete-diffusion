import torch
import torch.nn.functional as F
from fldd.blocks import compute_block_target


def compute_elbo_loss(model, forward_process, x, T, block_size=1):
    """Compute the discrete diffusion ELBO loss.

    Supports both pixel-factorized (block_size=1) and block-factorized
    (block_size=2,4) reverse models.

    The ELBO decomposes as:
        L = sum_{t=1}^{T} E_{z_t}[ KL[q(z_{t-1}|z_t,x) || p_theta(z_{t-1}|z_t)] ]
            + KL[q(z_T|x) || p(z_T)]

    For t=1, z_0 = x (data), so q(z_0|z_1,x) = delta(x).
    For t>1, q(z_s|z_t,x) = q(z_s|x) in the non-Markovian case.

    For block_size > 1, the target q(z_s^G | x) is a product of per-pixel
    Bernoullis (since the forward process is element-wise). The model predicts
    a full joint over block states. Over training, the model learns the
    data-averaged joint, which captures within-block correlations.
    """
    device = x.device
    B = x.shape[0]

    # sample a random timestep t uniformly from {1, ..., T}
    t = torch.randint(1, T + 1, (B,), device=device)

    # sample z_t from q(z_t | x)
    alphas = forward_process.get_alphas()
    alpha_t = alphas[t - 1]  # (B,)
    prob_one_zt = x * (1.0 - alpha_t[:, None, None, None]) + (1.0 - x) * alpha_t[:, None, None, None]
    z_t = torch.bernoulli(prob_one_zt)

    # model prediction
    logits = model(z_t, t - 1)  # 0-indexed timestep

    # compute per-pixel target probabilities for z_{t-1}
    is_first = (t == 1).float()[:, None, None, None]
    alpha_s = alphas[torch.clamp(t - 2, min=0)]
    target_pixel_prob = x * (1.0 - alpha_s[:, None, None, None]) + (1.0 - x) * alpha_s[:, None, None, None]
    # for t=1, target is delta(x)
    target_pixel_prob = is_first * x + (1.0 - is_first) * target_pixel_prob

    if block_size == 1:
        # pixel-factorized: binary cross-entropy
        pred_prob = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        bce = -(target_pixel_prob * torch.log(pred_prob)
                + (1 - target_pixel_prob) * torch.log(1 - pred_prob))
        reconstruction_loss = T * bce.sum(dim=(1, 2, 3)).mean()
    else:
        # block-factorized: cross-entropy over block categorical
        # target: product of Bernoullis -> (B, K^|G|, Hb, Wb)
        target_dist = compute_block_target(target_pixel_prob, block_size)

        # predicted: logits -> log_softmax over block states
        log_pred = F.log_softmax(logits, dim=1)

        # cross-entropy: -sum_s target(s) * log pred(s)
        ce = -(target_dist * log_pred).sum(dim=1)  # (B, Hb, Wb)
        reconstruction_loss = T * ce.sum(dim=(1, 2)).mean()

    # prior loss: KL[q(z_T|x) || Uniform]
    prior_loss = forward_process.kl_prior(x)

    loss = reconstruction_loss + prior_loss

    metrics = {
        "loss": loss.item(),
        "recon": reconstruction_loss.item(),
        "prior": prior_loss.item(),
    }

    return loss, metrics


def train_epoch(model, forward_process, train_loader, optimizer, T, device, block_size=1):
    model.train()
    forward_process.train()
    total_metrics = {"loss": 0, "recon": 0, "prior": 0}
    n_batches = 0

    for (x,) in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        loss, metrics = compute_elbo_loss(model, forward_process, x, T, block_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(forward_process.parameters()), 1.0
        )
        optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] += v
        n_batches += 1

    return {k: v / n_batches for k, v in total_metrics.items()}
