import torch
import torch.nn.functional as F


def compute_elbo_loss(model, forward_process, x, T):
    """Compute the discrete diffusion ELBO loss.

    The ELBO decomposes as:
        L = sum_{t=1}^{T} E_{z_t}[ KL[q(z_{t-1}|z_t,x) || p_theta(z_{t-1}|z_t)] ]
            + KL[q(z_T|x) || p(z_T)]

    For t=1, z_0 = x (data), so:
        KL[q(z_0|z_1,x) || p_theta(z_0|z_1)] = -E_{z_1}[log p_theta(x|z_1)]
        (since q(z_0|z_1,x) = delta(x))

    For t>1, q(z_{s}|z_t,x) = q(z_s|x) in the non-Markovian case.

    Args:
        model: UNet reverse model
        forward_process: LearnedForwardProcess
        x: (B, 1, 28, 28) binary data
        T: number of diffusion steps

    Returns:
        loss: scalar, negative ELBO (to minimize)
        metrics: dict with loss components
    """
    device = x.device
    B = x.shape[0]
    total_kl = 0.0

    # sample a random timestep t uniformly from {1, ..., T}
    # (importance-weighted single-step estimate instead of summing all)
    t = torch.randint(1, T + 1, (B,), device=device)

    # for each sample in the batch, sample z_t from q(z_t | x)
    # t is 1-indexed, so t_idx = t - 1
    alphas = forward_process.get_alphas()

    # gather alpha for each sample's timestep
    alpha_t = alphas[t - 1]  # (B,)
    prob_one_zt = x * (1.0 - alpha_t[:, None, None, None]) + (1.0 - x) * alpha_t[:, None, None, None]
    z_t = torch.bernoulli(prob_one_zt)

    # get model prediction: logits for p_theta(z_{t-1} | z_t)
    logits = model(z_t, t - 1)  # 0-indexed for the model
    pred_prob = torch.sigmoid(logits)

    # compute target: q(z_{t-1} | z_t, x)
    # for t=1: target is x itself (z_0 = x)
    # for t>1: target is q(z_{t-1} | x) = Bernoulli with prob from forward process
    is_first = (t == 1).float()[:, None, None, None]

    # target for t > 1
    alpha_s = alphas[torch.clamp(t - 2, min=0)]  # alpha at s = t-1, 0-indexed
    target_prob = x * (1.0 - alpha_s[:, None, None, None]) + (1.0 - x) * alpha_s[:, None, None, None]

    # for t=1, target is just x
    target_prob = is_first * x + (1.0 - is_first) * target_prob

    # binary cross-entropy loss (= KL up to a constant for Bernoulli)
    eps = 1e-7
    pred_prob = pred_prob.clamp(eps, 1 - eps)
    bce = -(target_prob * torch.log(pred_prob) + (1 - target_prob) * torch.log(1 - pred_prob))

    # multiply by T to account for uniform sampling of timestep
    reconstruction_loss = T * bce.sum(dim=(1, 2, 3)).mean()

    # prior loss: KL[q(z_T|x) || Uniform]
    prior_loss = forward_process.kl_prior(x)

    loss = reconstruction_loss + prior_loss

    metrics = {
        "loss": loss.item(),
        "recon": reconstruction_loss.item(),
        "prior": prior_loss.item(),
    }

    return loss, metrics


def train_epoch(model, forward_process, train_loader, optimizer, T, device):
    model.train()
    forward_process.train()
    total_metrics = {"loss": 0, "recon": 0, "prior": 0}
    n_batches = 0

    for (x,) in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        loss, metrics = compute_elbo_loss(model, forward_process, x, T)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(forward_process.parameters()), 1.0
        )
        optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] += v
        n_batches += 1

    return {k: v / n_batches for k, v in total_metrics.items()}
