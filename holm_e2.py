"""Holm-Bonferroni correction across the three E2 pairwise FID comparisons.

Family = {|G|=1 vs 2, |G|=1 vs 4, |G|=2 vs 4}. Reads the two-sided paired
t-test p-values from the three merged result files and applies the Holm
step-down procedure (uniformly more powerful than plain Bonferroni).
"""

import json

FILES = {
    "1v2": "results/results_e2_merged_1v2.json",
    "1v4": "results/results_e2_merged.json",
    "2v4": "results/results_e2_merged_2v4.json",
}


def main():
    raw = {}
    for name, path in FILES.items():
        with open(path) as f:
            raw[name] = float(json.load(f)["fid"]["t_test"]["p_two_sided"])

    # Holm step-down: sort ascending, adjusted p_i = max over j<=i of (m-j)*p_(j)
    items = sorted(raw.items(), key=lambda kv: kv[1])
    m = len(items)
    out = []
    running = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out.append({
            "comparison": name,
            "p_raw_two_sided": p,
            "p_holm": adj,
            "reject_at_0.05": adj < 0.05,
        })

    print(f"{'comparison':>10} {'p_raw':>10} {'p_holm':>10} {'reject@.05':>11}")
    for r in out:
        print(f"{r['comparison']:>10} {r['p_raw_two_sided']:>10.4f} "
              f"{r['p_holm']:>10.4f} {str(r['reject_at_0.05']):>11}")

    with open("results/results_e2_holm.json", "w") as f:
        json.dump({"family": list(FILES), "alpha": 0.05, "results": out}, f, indent=2)
    print("\nwrote results/results_e2_holm.json")


if __name__ == "__main__":
    main()
