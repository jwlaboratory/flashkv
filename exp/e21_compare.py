"""Paired comparison: mean-scaled vs marginal-scaled residency bonus.

Both runs use the same RNG seed and therefore the same samples, so every
comparison below is paired on (sample, budget).
"""
import collections, json, math
import numpy as np
A = json.load(open("results/e21_partial.json"))    # bonus="mean", incl. dense & top-k
B = json.load(open("results/e21_marg.json"))       # bonus="marginal"
key = lambda x: (x["sample"], x["budget"])
base = {key(x): x["score"] for x in A if x["lam"] == 0.0}
dense = {key(x): x["score"] for x in A if x["lam"] is None}
common = set(base)
print(f"mean-run samples: {len(set(x['sample'] for x in A))}, "
      f"marginal-run samples: {len(set(x['sample'] for x in B))}")
print(f"paired cells available: {len(common)}\n")
print(f"{'budget':>8}{'selector':>34}{'delta vs top-k':>18}{'n':>5}{'sig':>5}")
def ci(d): return 1.96 * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float('nan')
rows = []
for b in sorted(set(x["budget"] for x in A)):
    ks = [k for k in common if k[1] == b]
    d = np.array([dense[k] - base[k] for k in ks if k in dense])
    print(f"{b*100:7.2f}%{'dense (upper bound)':>34}{d.mean()*100:+13.2f}pp"
          f"{len(d):5d}{'*' if abs(d.mean())>ci(d) else 'ns':>5}")
    for src, lab in [(A, "mean-scaled"), (B, "marginal-scaled")]:
        for lam in [0.1, 0.3, 1.0]:
            cur = {key(x): x["score"] for x in src
                   if x["lam"] == lam and x.get("bonus", "mean") ==
                   ("mean" if src is A else "marginal")}
            ks2 = [k for k in ks if k in cur]
            if len(ks2) < 3: continue
            d = np.array([cur[k] - base[k] for k in ks2])
            c = ci(d)
            print(f"{'':8}{lab + f'  lam={lam}':>34}{d.mean()*100:+13.2f}pp"
                  f"{len(d):5d}{'*' if abs(d.mean())>c else 'ns':>5}")
            rows.append(dict(budget=b, bonus=lab, lam=lam, delta=float(d.mean()),
                             ci=float(c), n=len(d)))
    print()
json.dump(rows, open("results/e21_compare.json", "w"), indent=1)
print("delta = change in RULER score vs the standard block-sparse top-k selector,")
print("paired on identical samples. * = 95% CI excludes zero.")
