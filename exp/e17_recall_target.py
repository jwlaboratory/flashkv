"""E17: how accurate would the predictor have to be?

Speculation's real value is not hiding a round trip -- it is BREAKING THE SERIAL
CHAIN.  Layer l+1's query depends on layer l's output, so a reactive pager pays
up to 61 sequential round trips per token.  Predict all layers at once and a
correct prediction costs ZERO round trips.  But one miss anywhere in a layer
re-inserts that layer's round trip, so what matters is

    layers needing a corrective RTT = L x (1 - r^k)

with r the per-block recall and k=32 blocks per layer.  r=0.93 leaves 90% of
layers stalling; the chain barely shortens.  This sweeps r to find the target.
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
STEPS = 24
need, _, forced = synth_ws(L, nb, k, STEPS + 8, FRESH, 0.65, 0.76, 1, 4)

def pf_sets(r, mult=2, seed=0):
    rng = np.random.default_rng(seed)
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

def tpot(r, rtt, bw=12.5, ovh=3, mr=10):
    pf = pf_sets(r)
    return simulate("sparse_predicted", need[:STEPS], pf[0], L, nb, BLOCK*BPT, 0.0,
                    .025/L, Link(bw, ovh, mr), rtt_us=float(rtt), demand=True,
                    background=False, cold_unit=1, prefetch=pf, lookahead=1)["tpot"]*1000

def bulk(rtt, bw=12.5, ovh=3, mr=10):
    return simulate("layerwise", need[:STEPS], np.zeros((L, nb), bool), L, nb,
                    BLOCK*BPT, 0.0, .025/L, Link(bw, ovh, mr), rtt_us=float(rtt),
                    demand=True)["tpot"]*1000

print(f"128k, {L} layers, k={k} blocks/layer, 100GbE, prefix-cache hit. ideal TPOT 25.0 ms\n")
print(f"{'per-block':>10}{'layers still':>14}{'':>4}" +
      "".join(f"{'RTT '+str(r)+'us':>13}" for r in [100, 500, 1000, 5000]))
print(f"{'recall r':>10}{'stalling':>14}{'':>4}" + "".join(f"{'TPOT ms':>13}" for _ in range(4)))
print("-" * 82)
rows = []
RS = [0.90, 0.93, 0.959, 0.977, 0.99, 0.999, 0.9999, 1.0]
for r in RS:
    stall_layers = L * (1 - r**k)
    c = "".join(f"{tpot(r, rt):13.1f}" for rt in [100, 500, 1000, 5000])
    tag = ""
    if abs(r-0.93) < 1e-9: tag = "  <- measured, 2x over-fetch"
    if abs(r-0.959) < 1e-9: tag = "  <- measured, 4x over-fetch"
    if abs(r-0.977) < 1e-9: tag = "  <- measured, 8x over-fetch"
    print(f"{r:>10.4f}{stall_layers:>11.1f}/61{'':>4}{c}{tag}")
    rows.append(dict(recall=r, stall_layers=stall_layers,
                     **{f"rtt{rt}": tpot(r, rt) for rt in [100, 500, 1000, 5000]}))
print(f"{'bulk xfer':>10}{'-':>14}{'':>4}" +
      "".join(f"{bulk(rt):13.1f}" for rt in [100, 500, 1000, 5000]))

print("\nmax RTT at which paging still beats bulk transfer:")
for r in RS:
    lo, hi = 50, 60000
    for _ in range(18):
        mid = (lo * hi) ** 0.5
        if tpot(r, mid) < bulk(mid): lo = mid
        else: hi = mid
    print(f"  r={r:<8.4f} -> {lo/1000:7.2f} ms RTT"
          + ("   (cross-datacenter territory)" if lo > 2000 else ""))
json.dump(rows, open("results/e17_recall.json", "w"), indent=1)
print("\nwrote results/e17_recall.json")
