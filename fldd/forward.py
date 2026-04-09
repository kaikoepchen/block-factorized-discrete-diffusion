import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedForwardProcess(nn.Module):
    """Learned element-wise forward (noising) process for binary data.

    For each timestep t in {1, ..., T}, we learn a flip probability alpha_t
    such that:
        q_phi(z_t = x | x) = 1 - alpha_t
        q_phi(z_t != x | x) = alpha_t

    We parameterize via unconstrained logits and enforce monotonicity:
    alpha_1 <= alpha_2 <= ... <= alpha_T, with alpha_T close to 0.5 (uniform).

    The posterior q(z_s | z_t, x) is computed in closed form via Bayes' rule.
    """

    def __init__(self, T):
        super().__init__()
        self.T = T
        # unconstrained params; we'll softplus + cumsum to get monotone alphas
        self.logits = nn.Parameter(torch.zeros(T))

    def get_alphas(self):
        """Return flip probabilities alpha_1, ..., alpha_T in [0, 0.5).

        Monotonically increasing via cumulative softplus, then sigmoid
        to keep in (0, 0.5).
        """
        increments = F.softplus(self.logits)
        cumulative = torch.cumsum(increments, dim=0)
        # map to (0, 0.5) — sigmoid gives (0,1), multiply by 0.5
        alphas = 0.5 * torch.sigmoid(cumulative - 2.0)
        return alphas

    def q_zt_given_x(self, x, t_idx):
        """Compute q(z_t | x) for binary x.

        Args:
            x: binary tensor (B, 1, 28, 28), values in {0, 1}
            t_idx: integer timestep index (0-based, so t=1 is index 0)

        Returns:
            prob_one: probability that z_t = 1, shape (B, 1, 28, 28)
        """
        alphas = self.get_alphas()
        alpha_t = alphas[t_idx]

        # q(z_t = 1 | x) = x * (1 - alpha_t) + (1 - x) * alpha_t
        prob_one = x * (1.0 - alpha_t) + (1.0 - x) * alpha_t
        return prob_one

    def sample_zt(self, x, t_idx):
        """Sample z_t ~ q(z_t | x).

        Returns binary samples and the probabilities used.
        """
        prob_one = self.q_zt_given_x(x, t_idx)
        z_t = torch.bernoulli(prob_one)
        return z_t, prob_one

    def q_posterior(self, z_t, x, t_idx, s_idx):
        """Compute q(z_s | z_t, x) in closed form via Bayes' rule.

        q(z_s | z_t, x) = q(z_t | z_s) * q(z_s | x) / q(z_t | x)

        Since the forward process is element-wise and non-Markovian,
        q(z_t | z_s) is derived from the marginals:
            q(z_t | z_s, x) = q(z_t, z_s | x) / q(z_s | x)

        For binary case, q(z_t, z_s | x) can be computed from:
            q(z_t = j, z_s = i | x) = q(z_t = j | z_s = i, x) * q(z_s = i | x)

        We need q(z_t | z_s, x). Since both z_t and z_s are conditionally
        independent given x in the non-Markovian formulation, we have:
            q(z_t, z_s | x) = q(z_t | x) * q(z_s | x)

        Wait — that's only true if z_t and z_s are independent given x.
        In the non-Markovian FLDD formulation, they ARE independent given x
        because q(z_t | x) is defined directly (not through z_s).

        So the posterior becomes:
            q(z_s | z_t, x) = q(z_s | x)  [z_s indep of z_t given x]

        This is correct for the non-Markovian case! The posterior just
        depends on x, not on z_t. But then the reverse model p(z_s | z_t)
        has to learn q(z_s | x) averaged over x ~ q(x | z_t).

        Actually, let me reconsider. In FLDD the training target for the
        reverse model at step t->s is:
            q_phi(z_s | z_t, x) — the posterior given a specific x

        For a Markovian forward chain, this is non-trivial. For the
        non-Markovian case with independent marginals, this equals q(z_s | x).

        But the LOSS is:
            E_{x, z_t} [ KL[ q(z_s | z_t, x) || p_theta(z_s | z_t) ] ]

        With q(z_s | z_t, x) = q(z_s | x) (indep of z_t given x), the
        reverse model still needs z_t as input because different z_t values
        correspond to different posterior distributions over x.

        Returns:
            prob_one: q(z_s = 1 | z_t, x), shape same as x
        """
        # In non-Markovian FLDD: q(z_s | z_t, x) = q(z_s | x)
        prob_one = self.q_zt_given_x(x, s_idx)
        return prob_one

    def kl_prior(self, x):
        """KL divergence at the final step: KL[q(z_T | x) || p(z_T)].

        p(z_T) is uniform Bernoulli(0.5).
        """
        prob_one = self.q_zt_given_x(x, self.T - 1)
        # KL[Bernoulli(p) || Bernoulli(0.5)]
        # = p*log(2p) + (1-p)*log(2(1-p))
        eps = 1e-8
        p = prob_one.clamp(eps, 1 - eps)
        kl = p * torch.log(2.0 * p) + (1.0 - p) * torch.log(2.0 * (1.0 - p))
        return kl.sum(dim=(1, 2, 3)).mean()
