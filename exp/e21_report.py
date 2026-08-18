"""Summarise the RULER eval: accuracy by task, budget and selector."""
import collections, json, math, sys
import numpy as np
r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/e21_partial.json"))
print(f"{len(r)} rows, {len(set(x['sample'] for x in r))} samples, "
      f"ctx~{r[0]['ctx']}, k={sorted(set(x['k'] for x in r))} of {r[0]['nb']} blocks")
def name(l): return "dense" if l is None else ("block-sparse top-k" if l == 0
                                               else f"cache-aware lam={l}")
for task in ["multikey", "multivalue"]:
    sub = [x for x in r if x.get("task") == task]
    if not sub: continue
    print(f"\n=== {task} ===")
    print(f"{'budget':>8}{'selector':>22}{'exact':>9}{'partial':>10}{'n':>5}{'fresh':>9}")
    g = collections.defaultdict(list)
    for x in sub: g[(x["budget"], x["lam"])].append(x)
    for kk in sorted(g, key=lambda t: (t[0], -1 if t[1] is None else t[1])):
        v = g[kk]
        ex = np.mean([x["correct"] for x in v]) * 100
        pa = np.mean([x["score"] for x in v]) * 100
        fr = np.mean([x["fresh"] for x in v])
        print(f"{kk[0]*100:7.2f}%{name(kk[1]):>22}{ex:8.1f}%{pa:9.1f}%{len(v):5d}{fr:9.3f}")
# paired comparison vs top-k at each budget
print("\n=== paired vs block-sparse top-k (same samples) ===")
for b in sorted(set(x["budget"] for x in r)):
    base = {x["sample"]: x["score"] for x in r if x["budget"] == b and x["lam"] == 0.0}
    print(f"  budget {b*100:.2f}%")
    for l in [None, 0.1, 0.3, 1.0]:
        cur = {x["sample"]: x["score"] for x in r if x["budget"] == b and x["lam"] == l}
        both = sorted(set(base) & set(cur))
        if not both: continue
        d = np.array([cur[s] - base[s] for s in both])
        ci = 1.96 * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        sig = "*" if abs(d.mean()) > ci else "ns"
        print(f"    {name(l):<22}{d.mean()*100:+7.2f}pp +-{ci*100:5.2f}  n={len(d)}  {sig}")
