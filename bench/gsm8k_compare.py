#!/usr/bin/env python3
"""Paired comparison of gsm8k_eval.py result files (same questions, same order): accuracy, SE, discordant pairs, exact McNemar p.
Usage: gsm8k_compare.py REF.json OTHER.json [OTHER2.json ...]   (first file = reference; per-question pairing by index `i`)
"""
import json, math, sys


def load(p):
    d = json.load(open(p))
    return d["summary"], {r["i"]: r for r in d["results"]}


def binom_two_sided(k, n):
    """Exact two-sided p for k successes out of n under p=0.5 (McNemar on discordant pairs)."""
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    p = sum(math.comb(n, j) for j in range(0, lo + 1)) / 2 ** n
    return min(1.0, 2 * p)


ref_s, ref = load(sys.argv[1])
n = ref_s["n"]
acc = ref_s["accuracy"]
print(f"ref   {sys.argv[1]}: acc={acc:.4f} (SE ±{math.sqrt(acc * (1 - acc) / n):.4f}) n={n} truncated={ref_s['truncated']} errors={ref_s['errors']} "
      f"tok/s={ref_s['tok_s']} completion_tokens={ref_s['completion_tokens']}")
for p in sys.argv[2:]:
    s, other = load(p)
    common = sorted(set(ref) & set(other))
    a_only = sum(1 for i in common if ref[i]["ok"] and not other[i]["ok"])   # ref right, other wrong
    b_only = sum(1 for i in common if other[i]["ok"] and not ref[i]["ok"])   # other right, ref wrong
    both_wrong = sum(1 for i in common if not ref[i]["ok"] and not other[i]["ok"])
    same_pred = sum(1 for i in common if ref[i].get("pred") == other[i].get("pred"))
    acc_o = s["accuracy"]
    print(f"other {p}: acc={acc_o:.4f} (SE ±{math.sqrt(acc_o * (1 - acc_o) / s['n']):.4f}) n={s['n']} truncated={s['truncated']} errors={s['errors']} "
          f"tok/s={s['tok_s']} completion_tokens={s['completion_tokens']}")
    print(f"      paired on {len(common)}: delta={acc_o - acc:+.4f}  ref-only-right={a_only}  other-only-right={b_only}  both-wrong={both_wrong}  "
          f"same-pred={same_pred}  McNemar exact p={binom_two_sided(a_only, a_only + b_only):.3f}")
