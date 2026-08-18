"""E10: does the working set saturate, or does a long generation touch everything?

If it saturates low, the cold majority of the KV never needs to move at all and
"priority order + lazy background stream" is dominated by "never send the cold
blocks".  If it climbs to 100%, paging is a bandwidth disaster and the bulk
stream is the right design.  16 decode steps cannot tell these apart; 256 can.
"""
import glob, json, os
import numpy as np

def sel_shared(sc, k):
    agg = sc.sum(0); m = np.zeros(len(agg), bool)
    m[np.argpartition(-agg, min(k, len(agg)-1))[:k]] = True
    return m

def curve(path, budget_frac=0.0156):
    z = np.load(path + ".npz"); meta = json.load(open(path + ".json"))
    sc = z["scores"].astype(np.float32); T, L, H, nb = sc.shape
    ntail = meta.get("n_tail", 1); k = max(1, int(round(budget_frac * nb)))
    cum = np.zeros((L, nb), bool)
    ys, fresh = [], []
    for t in range(ntail, T):
        new = 0
        for l in range(L):
            m = sel_shared(sc[t, l], k)
            new += int((m & ~cum[l]).sum()); cum[l] |= m
        ys.append(cum.mean()); fresh.append(new / (L * k))
    return dict(trace=os.path.basename(path), ctx=meta["ctx"], L=L, nb=nb, k=k,
                steps=len(ys), curve=[float(v) for v in ys],
                fresh_rate=[float(v) for v in fresh])

if __name__ == "__main__":
    paths = sorted(p[:-4] for p in glob.glob("results/e1l/*.npz"))
    out = [curve(p) for p in paths]
    print(f"{'trace':<20}{'ctx':>7}{'blocks':>8}" +
          "".join(f"{'WS@'+str(s):>9}" for s in [1, 8, 32, 64, 128, 256]) +
          f"{'fresh/step last 32':>21}")
    for r in out:
        c = r["curve"]
        cells = "".join(f"{c[min(s, len(c)) - 1] * 100:8.1f}%" for s in [1, 8, 32, 64, 128, 256])
        print(f"{r['trace'][:19]:<20}{r['ctx']:>7}{r['nb']:>8}{cells}"
              f"{np.mean(r['fresh_rate'][-32:]) * 100:19.1f}%")
    json.dump(out, open("results/e10_longgen.json", "w"), indent=1)
    print("\n'fresh/step' = share of each step's selected blocks never seen before.")
    print("At steady state, bytes/step for pure paging = fresh_rate * budget.")
    print("wrote results/e10_longgen.json")
