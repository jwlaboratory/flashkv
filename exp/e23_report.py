"""Report E23: cache-aware selection on MiniCPM4.1-8B (trained sparse attention)."""
import collections, json, math, sys
import numpy as np
r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/e23_mcpm.json"))
ns = len(set(x["sample"] for x in r))
print(f"{len(r)} rows, {ns} samples, ctx~{r[0]['ctx']}, {r[0]['nb']} blocks\n")
def nm(l, b): return ("dense" if l is None else
                      "block-sparse top-k" if l == 0 else f"{b} lam={l}")
print(f"{'k':>4}{'selector':>24}{'exact':>8}{'partial':>9}{'n':>4}{'fresh':>8}")
g = collections.defaultdict(list)
for x in r: g[(x["k"], x["lam"], x["bonus"] if x["lam"] else "")].append(x)
for key in sorted(g, key=lambda t: (t[0], -1 if t[1] is None else t[1], t[2])):
    v = g[key]
    print(f"{key[0]:>4}{nm(key[1], key[2]):>24}"
          f"{np.mean([x['correct'] for x in v])*100:7.1f}%"
          f"{np.mean([x['score'] for x in v])*100:8.1f}%{len(v):4d}"
          f"{np.mean([x['fresh'] for x in v]):8.3f}")
print("\n=== paired vs block-sparse top-k (same samples, same k) ===")
def ci(d): return 1.96*d.std(ddof=1)/math.sqrt(len(d)) if len(d) > 1 else float('nan')
for k in sorted(set(x["k"] for x in r)):
    base = {x["sample"]: x["score"] for x in r if x["k"] == k and x["lam"] == 0.0}
    print(f"  k={k} blocks ({k/r[0]['nb']*100:.2f}% of cache)")
    dn = {x["sample"]: x["score"] for x in r if x["lam"] is None}
    if dn:
        ks = sorted(set(base) & set(dn))
        d = np.array([dn[s]-base[s] for s in ks])
        if len(d) > 1:
            print(f"    {'dense (upper bound)':<26}{d.mean()*100:+7.2f}pp "
                  f"+-{ci(d)*100:5.2f} n={len(d)} {'*' if abs(d.mean())>ci(d) else 'ns'}")
    for b in ["mean", "marginal"]:
        for lam in [0.1, 0.3, 1.0]:
            cur = {x["sample"]: x["score"] for x in r
                   if x["k"] == k and x["lam"] == lam and x["bonus"] == b}
            ks = sorted(set(base) & set(cur))
            if len(ks) < 2: continue
            d = np.array([cur[s]-base[s] for s in ks]); c = ci(d)
            print(f"    {b+f' lam={lam}':<26}{d.mean()*100:+7.2f}pp +-{c*100:5.2f}"
                  f" n={len(d)} {'*' if abs(d.mean())>c else 'ns'}")
