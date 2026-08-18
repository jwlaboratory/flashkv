"""E8: how much should we speculatively over-fetch?

Over-fetching trades bytes for misses.  At a DSA budget of 1.56% of the cache,
sending 4x the budget is still only 6% -- if that buys 94% recall, nearly every
demand pull disappears.  Recall values are the MEASURED last-prefill-token
numbers from E7; the simulator says what they are worth in TTFT.
"""
import json, sys
import numpy as np
sys.path.insert(0, "exp")
from e2_pipeline_sim import Link, simulate, synth

L, BLOCK, BPT, CTX, BUDGET = 61, 64, 1152.0, 131072, 2048
nb, k = CTX // BLOCK, BUDGET // BLOCK

# measured in E7: last_prefill predictor, first decode token, 1.56% budget
RECALL = {1: 0.750, 2: 0.868, 3: 0.913, 4: 0.938, 6: 0.960, 8: 0.979}

def pred_mask(need0, m, rec, rng):
    """priority set of m*k blocks containing rec*k of the truly-needed ones"""
    P = np.zeros_like(need0)
    for l in range(L):
        hit = np.flatnonzero(need0[l])
        keep = rng.choice(hit, int(round(rec * len(hit))), replace=False)
        P[l, keep] = True
        need_more = m * k - P[l].sum()
        cold = np.flatnonzero(~P[l])
        if need_more > 0:
            P[l, rng.choice(cold, min(int(need_more), len(cold)), replace=False)] = True
    return P

SEEDS = 6
rng = np.random.default_rng(0)
need, _, forced = synth(L, nb, k, 12, 0.65, 0.76, 1, 4)
print(f"ctx={CTX} L={L} KV={L*nb*BLOCK*BPT/1e9:.2f} GB  budget={k} blocks "
      f"({k/nb*100:.2f}% of cache)  prefix-cache-hit scenario\n")
rows = []
for lname, bw, ovh, mr in [("100GbE", 12.5, 3, 10), ("IB NDR 400Gb", 50, 1.5, 20),
                           ("25GbE x-DC", 3.1, 10, 5)]:
    link = Link(bw, ovh, mr)
    print(f"--- {lname} ---")
    print(f"{'over-fetch':>11}{'prio set':>10}{'recall':>8}{'TTFT ms':>10}"
          f"{'MB by TTFT':>12}{'post-TTFT stall':>17}{'demand pulls':>14}")
    base = simulate("layerwise", need, np.zeros((L, nb), bool), L, nb, BLOCK*BPT,
                    0.0, .025/L, link, rtt_us=20., demand=True)
    print(f"{'layer-wise':>11}{'-':>10}{'-':>8}{base['ttft']*1000:10.1f}"
          f"{base['bytes_before_ttft']/1e6:12.0f}{base['stall_post_ttft']*1000:16.1f}ms")
    for m, rec in RECALL.items():
        tt, mb, st, pf = [], [], [], []
        for seed in range(SEEDS):
            P = pred_mask(need[0], m, rec, np.random.default_rng(seed))
            r = simulate("sparse_predicted", need, P, L, nb, BLOCK*BPT, 0.0, .025/L,
                         link, rtt_us=20., demand=True)
            tt.append(r["ttft"]*1000); mb.append(r["bytes_before_ttft"]/1e6)
            st.append(r["stall_post_ttft"]*1000); pf.append(float(P.mean()))
        misses = int(round((1 - rec) * k * L))
        print(f"{m:10d}x{np.mean(pf)*100:9.2f}%{rec:8.3f}"
              f"{np.mean(tt):8.1f}+-{np.std(tt):3.0f}{np.mean(mb):12.0f}"
              f"{np.mean(st):14.1f}ms{misses:14d}")
        rows.append(dict(link=lname, m=m, recall=rec, ttft_ms=float(np.mean(tt)),
                         ttft_sd=float(np.std(tt)), prio_frac=float(np.mean(pf)),
                         stall=float(np.mean(st)), ttft_layerwise=base["ttft"]*1000))
    print()
json.dump(rows, open("results/e8_overfetch.json", "w"), indent=1)
print("wrote results/e8_overfetch.json")
