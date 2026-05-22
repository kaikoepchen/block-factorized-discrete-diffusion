"""Sanity check for the reparameterized forward schedule.

Verifies that the new alpha_t = 0.5*(1 - exp(-cumsum(softplus(logits)))) form
(1) has no structural floor, (2) starts from a sensible monotone init, and
(3) can be driven both UP and DOWN by gradient descent on the schedule logits.
"""
import torch
from fldd.forward import LearnedForwardProcess

torch.manual_seed(0)
T = 4
fp = LearnedForwardProcess(T)

print(f"init alphas (logits=0): {[round(a, 4) for a in fp.get_alphas().tolist()]}")
print(f"floor as logits->-inf : "
      f"{[round(a, 6) for a in (0.5*(1-torch.exp(-torch.cumsum(torch.nn.functional.softplus(torch.full((T,), -50.0)), 0)))).tolist()]}")

# Drive the schedule DOWN (target all-zero) and UP (target all-0.5).
for target_val, name in [(0.0, "DOWN->0"), (0.49, "UP->0.49")]:
    fp = LearnedForwardProcess(T)
    opt = torch.optim.Adam(fp.parameters(), lr=0.1)
    target = torch.full((T,), target_val)
    for _ in range(2000):
        opt.zero_grad()
        loss = ((fp.get_alphas() - target) ** 2).mean()
        loss.backward()
        opt.step()
    a = fp.get_alphas()
    print(f"{name:>10}: alphas={[round(v, 4) for v in a.tolist()]} "
          f"(monotone={bool((a[1:] >= a[:-1]).all())})")
