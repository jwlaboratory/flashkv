"""E15: how far AHEAD can we predict?

Reactive paging discovers it needs a block at the moment it needs it, so the
round trip lands on the critical path and only ~0.4ms of layer compute hides it.
If step t's selection predicts step t+h, you can issue the fetch h token-times
early -- ~25ms of slack per step instead of 0.4ms.  Whether that works is purely
a question of how fast recall decays with horizon.

Predictors evaluated at horizon h:
  last        step t's selection alone
  recent_u    union of the last 4 steps (a "recent working set")
  cumulative  every block seen so far this generation (the LRU-style set)
"""
import glob, json, os
import numpy as np

def topk(v, n):
    m = np.zeros(len(v), bool); m[np.argpartition(-v, min(n, len(v)-1))[:n]] = True
    return m

def analyze(path, budget_frac=0.0156, H=(1, 2, 4, 8, 16), M=(1, 2, 4)):
    z = np.load(path + ".npz"); meta = json.load(open(path + ".json"))
    sc = z["scores"].astype(np.float32); T, L, H_, nb = sc.shape
    nt = meta.get("n_tail", 1); k = max(1, int(round(budget_frac * nb)))
    agg = sc.sum(2)                                   # [T, L, nb] shared-index score
    ns = T - nt
    truth = np.zeros((ns, L, nb), bool)
    for d in range(ns):
        for l in range(L):
            truth[d, l] = topk(agg[nt + d, l], k)
    out = dict(trace=os.path.basename(path), ctx=meta["ctx"], L=L, nb=nb, k=k, res={})
    for h in H:
        for m in M:
            n = int(m * k)
            r_last, r_recent, r_cum = [], [], []
            for d in range(4, ns - h):
                for l in range(0, L, 3):              # subsample layers for speed
                    tgt = truth[d + h, l]
                    # last: rank by step d's score
                    r_last.append((topk(agg[nt + d, l], n) & tgt).sum() / max(1, tgt.sum()))
                    # recent union of last 4 steps, ranked by summed score
                    rec = agg[nt + d - 3:nt + d + 1, l].sum(0)
                    r_recent.append((topk(rec, n) & tgt).sum() / max(1, tgt.sum()))
                    # cumulative: everything selected so far (set, not ranking)
                    cum = truth[:d + 1, l].any(0)
                    r_cum.append((cum & tgt).sum() / max(1, tgt.sum()))
            out["res"][f"h{h}_m{m}"] = dict(
                h=h, m=m, last=float(np.mean(r_last)),
                recent4=float(np.mean(r_recent)), cumulative=float(np.mean(r_cum)),
                cum_size=float(np.mean([truth[:d+1].any(0).mean()
                                        for d in range(4, ns - h, 8)])))
    return out

if __name__ == "__main__":
    paths = sorted(p[:-4] for p in glob.glob("results/e1l/*.npz"))
    A = [analyze(p) for p in paths]
    print(f"{len(A)} traces, 256-step generations, DSA-proportional budget\n")
    print("recall of step t+h's needed blocks, predicted at step t")
    print(f"{'horizon':>8}{'slack':>10}" +
          "".join(f"{'m='+str(m):>26}" for m in (1, 2, 4)))
    print(f"{'':>18}" + "".join(f"{'last / recent4':>26}" for _ in range(3)))
    for h in (1, 2, 4, 8, 16):
        c = ""
        for m in (1, 2, 4):
            kk = f"h{h}_m{m}"
            l = np.mean([a["res"][kk]["last"] for a in A])
            r = np.mean([a["res"][kk]["recent4"] for a in A])
            c += f"{l:14.3f} /{r:9.3f}  "
        print(f"{h:>8}{h*25:>7}ms {c}")
    cu = np.mean([A[0]["res"]["h1_m1"]["cumulative"]])
    sz = np.mean([a["res"]["h1_m1"]["cum_size"] for a in A])
    print(f"\ncumulative 'everything seen so far' set: recall "
          f"{np.mean([a['res']['h1_m1']['cumulative'] for a in A]):.3f} "
          f"at {sz*100:.1f}% of cache resident (this is what LRU-style caching buys)")
    json.dump(A, open("results/e15_horizon.json", "w"), indent=1)
    print("wrote results/e15_horizon.json")
