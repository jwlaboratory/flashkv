"""E16: does pipelined prefetch move the latency wall?

E11 found reactive paging loses to bulk transfer above ~100us RTT, because a
miss costs a round trip on the critical path, up to once per layer per token.
E15 found step t's selection predicts step t+h at 0.82-0.93 recall even at
h=16.  So issue the fetch h steps early and the round trip has h token-times to
hide in.  If that works the wall should move by orders of magnitude.
"""
import json, sys
import numpy as np
sys.path.insert(0, "exp")
from e2_pipeline_sim import Link, simulate, synth_ws

L, BLOCK, BPT = 61, 64, 1152.0
nb, k = 131072 // BLOCK, 2048 // BLOCK
C = json.load(open("results/e10_longgen.json"))
n = min(len(c["fresh_rate"]) for c in C)
FRESH = np.mean([c["fresh_rate"][:n] for c in C], axis=0)
STEPS = 32
need, pred0, forced = synth_ws(L, nb, k, STEPS + 20, FRESH, 0.65, 0.76, 1, 4)

# E15-measured recall of the horizon-h predictor at 2x over-fetch
REC = {1: 0.930, 2: 0.904, 4: 0.878, 8: 0.854, 16: 0.819}

def pf_sets(h, mult=2, seed=0):
    """predicted need-sets at horizon h: `mult`*k blocks containing REC[h] of truth"""
    rng = np.random.default_rng(seed)
    r = REC[h]
    out = []
    for t in range(len(need)):
        P = np.zeros((L, nb), bool)
        for l in range(L):
            hit = np.flatnonzero(need[t][l])
            keep = rng.choice(hit, int(round(r * len(hit))), replace=False)
            P[l, keep] = True
            cold = np.flatnonzero(~P[l])
            extra = mult * k - P[l].sum()
            if extra > 0:
                P[l, rng.choice(cold, min(int(extra), len(cold)), replace=False)] = True
        out.append(P)
    return out

print(f"128k ctx, DeepSeek geometry, {STEPS} decode steps, prefix-cache hit, 100GbE")
print("ideal TPOT (no transfer cost) = 25.0 ms\n")
print(f"{'RTT':>8}{'bulk TPOT':>12}" +
      "".join(f"{'paged h='+str(h):>16}" for h in [0, 1, 2, 4, 16]))
rows = []
for rtt in [20, 100, 500, 1000, 5000, 20000]:
    link = Link(12.5, 3, 10)
    a = dict(rtt_us=float(rtt), demand=True)
    bulk = simulate("layerwise", need[:STEPS], np.zeros((L, nb), bool), L, nb,
                    BLOCK*BPT, 0.0, .025/L, link, **a)["tpot"] * 1000
    c = ""
    rec = dict(rtt_us=rtt, bulk=bulk)
    for h in [0, 1, 2, 4, 16]:
        pf = pf_sets(max(1, h))
        r = simulate("sparse_predicted", need[:STEPS], pf[0], L, nb, BLOCK*BPT,
                     0.0, .025/L, link, background=False, cold_unit=1,
                     prefetch=pf if h else None, lookahead=h, **a)
        c += f"{r['tpot']*1000:15.1f} "
        rec[f"h{h}"] = r["tpot"] * 1000
    print(f"{rtt:6d}us{bulk:11.1f} {c}")
    rows.append(rec)
json.dump(rows, open("results/e16_lookahead.json", "w"), indent=1)
print("\nh=0 is reactive paging (what SAC does). h>=1 issues the fetch h steps early.")
print("wrote results/e16_lookahead.json")
