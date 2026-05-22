"""Recompute true ELBO from stored schedules by removing the parasitic H[q].

The pre-fix training loss used cross-entropy = KL + H[target]. H[target]
depends only on the forward alphas (not the reverse model), so it can be
subtracted analytically without retraining:

    parasitic = 784 * sum_{j=0}^{T-2} H_b(alpha_j)      (nats per image)

H_b is the binary entropy. Block size does not matter: the entropy of a product
distribution is the sum of its marginal entropies, and there are 784 pixels
either way. (t=1 target is delta(x), so its entropy is 0 and it is excluded.)

    corrected ELBO = reported loss - parasitic
"""

import json
import math

N_PIX = 784


def hb(a):
    a = min(max(a, 1e-12), 1 - 1e-12)
    return -(a * math.log(a) + (1 - a) * math.log(1 - a))


def parasitic(alphas):
    T = len(alphas)
    return N_PIX * sum(hb(alphas[j]) for j in range(0, T - 1))


def add_losses(loss, path, T_fixed=None):
    with open(path) as f:
        d = json.load(f)
    for r in d.get("per_run", []):
        T = r.get("T", T_fixed)
        L = r.get("best_loss", r.get("loss"))
        if T is None or L is None:
            continue
        loss[(int(T), int(r["block_size"]), int(r["seed"]))] = float(L)


def main():
    loss = {}
    add_losses(loss, "results/results_e4.json")
    add_losses(loss, "results/results_e2_merged.json", T_fixed=4)
    add_losses(loss, "results/results_e2_bs2.json", T_fixed=4)

    with open("results/schedule_summary.json") as f:
        sched = json.load(f)

    seen, rows = set(), []
    for e in sched["per_ckpt_e2"] + sched["per_ckpt_e4"]:
        key = (e["T"], e["block_size"], e["seed"])
        if key in seen:
            continue
        seen.add(key)
        par = parasitic(e["alphas"])
        L = loss.get(key)
        rows.append({
            "T": e["T"], "block_size": e["block_size"], "seed": e["seed"],
            "parasitic_Hq": par,
            "reported_loss": L,
            "corrected_elbo": (L - par) if L is not None else None,
            "corrected_elbo_per_step": ((L - par) / e["T"]) if L is not None else None,
            "Hq_fraction": (par / L) if L else None,
        })

    with open("results/results_elbo_corrected.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)

    hdr = (f"{'T':>3} {'bs':>3} {'seed':>4} {'reported':>10} {'parasitic':>10} "
           f"{'corrected':>10} {'per-step':>9} {'H[q]%':>6}")
    print(hdr)
    for r in sorted(rows, key=lambda x: (x["T"], x["block_size"], x["seed"])):
        if r["reported_loss"] is None:
            continue
        print(f"{r['T']:>3} {r['block_size']:>3} {r['seed']:>4} "
              f"{r['reported_loss']:>10.2f} {r['parasitic_Hq']:>10.2f} "
              f"{r['corrected_elbo']:>10.2f} {r['corrected_elbo_per_step']:>9.2f} "
              f"{100 * r['Hq_fraction']:>5.1f}%")
    print("\nwrote results/results_elbo_corrected.json")


if __name__ == "__main__":
    main()
