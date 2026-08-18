"""E5: regime map -- where does sparse-priority actually beat layer-wise?

Layer-wise pipelining already hides transfer behind prefill compute.  The only
place sparse ordering can win is where transfer is genuinely on the critical
path.  Two axes decide that: how much prefill compute there is to hide behind,
and how much per-request bandwidth the link gives you.

Reported as ABSOLUTE ms saved as well as ratio -- a 5x speedup on 20 ms is
not a product.
"""
import itertools, json, sys
import numpy as np
sys.path.insert(0, "exp")
from e2_pipeline_sim import Link, simulate, synth, LINKS

L, BLOCK, BPT = 61, 64, 1152.0
STEPS, DEC_MS = 16, 25.0

def run(ctx, budget_tok, bw, ovh, mr, pf_s, jac, pred, steps=STEPS, seed=0,
        demand=True):
    nb, k = ctx // BLOCK, budget_tok // BLOCK
    need, pmask, forced = synth(L, nb, k, steps, jac, pred, 1, max(1, 256 // BLOCK), seed)
    link = Link(bw, ovh, mr)
    args = (need, None, L, nb, BLOCK * BPT, pf_s / L, DEC_MS / 1000 / L, link)
    kw = dict(rtt_us=20.0, demand=demand)
    out = {}
    for pol, m in [("layerwise", np.zeros((L, nb), bool)),
                   ("sparse_oracle", need[0]),
                   ("sparse_predicted", pmask),
                   ("sparse_sinklocal", np.tile(forced, (L, 1)))]:
        a = list(args); a[1] = m
        out[pol] = simulate(pol, *a, **kw)
    return out

def pct(a, b): return (b - a) / b * 100 if b else 0.0

print("=" * 100)
print("A. REGIME MAP  -- TTFT reduction of sparse_predicted vs layerwise")
print("   ctx=128k, DSA budget=2048 tok (1.56%), jaccard=0.65, pred_hit=0.76")
print("=" * 100)
PF = [("cache hit (0 s)", 0.0), ("fast pf 20k tok/s", 131072 / 20000),
      ("typ pf 5k tok/s", 131072 / 5000), ("slow pf 1k tok/s", 131072 / 1000)]
BWS = [("25GbE x-DC", 3.1, 10, 5), ("100GbE", 12.5, 3, 10), ("RoCE 200Gb", 25, 2, 15),
       ("IB NDR 400Gb", 50, 1.5, 20), ("NVLink", 450, .5, 50)]
hdr = f"{'prefill scenario':<20}" + "".join(f"{n:>19}" for n, *_ in BWS)
print(hdr); print("-" * len(hdr))
rows = []
for pn, pf in PF:
    cells = ""
    for bn, bw, ovh, mr in BWS:
        r = run(131072, 2048, bw, ovh, mr, pf, 0.65, 0.76)
        base, sp = r["layerwise"]["ttft"], r["sparse_predicted"]["ttft"]
        saved = (base - sp) * 1000
        cells += f"{saved:11.1f}ms{pct(sp, base):6.1f}%"
        rows.append(dict(prefill=pn, link=bn, saved_ms=saved, pct=pct(sp, base),
                         ttft_layerwise=base * 1000, ttft_sparse=sp * 1000))
    print(f"{pn:<20}{cells}")

print("\n" + "=" * 100)
print("B. CONTEXT LENGTH  -- prefix-cache-hit scenario (no prefill to hide behind)")
print("   budget held at DSA's 2048 tokens, so the sparse fraction shrinks with ctx")
print("=" * 100)
print(f"{'ctx':>8}{'KV GB':>8}{'budget%':>9}{'link':>15}"
      f"{'TTFT layerwise':>16}{'TTFT sparse':>13}{'saved':>10}{'reduction':>11}")
for ctx in [8192, 32768, 131072, 524288]:
    for bn, bw, ovh, mr in [("100GbE", 12.5, 3, 10), ("IB NDR 400Gb", 50, 1.5, 20)]:
        r = run(ctx, 2048, bw, ovh, mr, 0.0, 0.65, 0.76)
        base, sp = r["layerwise"]["ttft"] * 1000, r["sparse_predicted"]["ttft"] * 1000
        gb = L * ctx * BPT / 1e9
        print(f"{ctx:>8}{gb:8.2f}{2048/ctx*100:8.2f}%{bn:>15}"
              f"{base:15.1f}ms{sp:12.1f}ms{base-sp:9.1f}ms{pct(sp/1000, base/1000):10.1f}%")
        rows.append(dict(ctx=ctx, link=bn, ttft_layerwise=base, ttft_sparse=sp))

print("\n" + "=" * 100)
print("C. SENSITIVITY  -- how good must the selector's stability / predictability be?")
print("   ctx=128k, 100GbE, prefix-cache hit.  TTFT ms and post-TTFT stall ms.")
print("=" * 100)
print(f"{'jaccard':>9}{'pred_hit':>10}" +
      "".join(f"{p:>26}" for p in ["sparse_predicted", "sparse_oracle"]))
for jac in [0.3, 0.5, 0.65, 0.85]:
    for ph in [0.5, 0.76, 1.0]:
        r = run(131072, 2048, 12.5, 3, 10, 0.0, jac, ph)
        c = ""
        for p in ["sparse_predicted", "sparse_oracle"]:
            c += f"{r[p]['ttft']*1000:14.1f} +{r[p]['stall_post_ttft']*1000:8.1f} stall"
        print(f"{jac:9.2f}{ph:10.2f}{c}")

print("\n" + "=" * 100)
print("D. BUDGET  -- what if the sparse budget is larger than DSA's 2048?")
print("   ctx=128k, 100GbE, prefix-cache hit")
print("=" * 100)
print(f"{'budget tok':>11}{'% of cache':>12}{'TTFT layerwise':>16}{'TTFT sparse':>14}{'reduction':>11}")
for bt in [1024, 2048, 4096, 8192, 16384, 32768]:
    r = run(131072, bt, 12.5, 3, 10, 0.0, 0.65, 0.76)
    base, sp = r["layerwise"]["ttft"] * 1000, r["sparse_predicted"]["ttft"] * 1000
    print(f"{bt:>11}{bt/131072*100:11.2f}%{base:15.1f}ms{sp:13.1f}ms"
          f"{pct(sp, base):10.1f}%")

json.dump(rows, open("results/e5_regime.json", "w"), indent=1)
print("\nwrote results/e5_regime.json")

print("\n" + "=" * 100)
print("E. PUSH-ONLY vs PUSH+DEMAND-PULL  -- ctx=128k, prefix-cache hit")
print("   push-only: receiver cannot reorder, it just waits for the stream.")
print("   This is the honest layer-wise baseline; demand-pull costs 20us RTT/fetch.")
print("=" * 100)
print(f"{'link':<15}{'mode':<12}" + "".join(f"{p:>20}" for p in
      ["layerwise", "sparse_predicted", "sparse_oracle"]))
for bn, bw, ovh, mr in [("100GbE", 12.5, 3, 10), ("IB NDR 400Gb", 50, 1.5, 20),
                        ("25GbE x-DC", 3.1, 10, 5)]:
    for mode, dm in [("push-only", False), ("push+pull", True)]:
        r = run(131072, 2048, bw, ovh, mr, 0.0, 0.65, 0.76, demand=dm)
        c = "".join(f"{r[p]['ttft']*1000:14.1f}ms   " for p in
                    ["layerwise", "sparse_predicted", "sparse_oracle"])
        print(f"{bn:<15}{mode:<12}{c}")

print("\n" + "=" * 100)
print("F. CROSSOVER  -- how low must per-request bandwidth go before sparse")
print("   ordering matters even in the FRESH-PREFILL case? (ctx=128k, 5k tok/s)")
print("=" * 100)
print(f"{'eff BW GB/s':>12}{'per-layer xfer ms':>19}{'per-layer prefill ms':>22}"
      f"{'TTFT layerwise':>16}{'TTFT sparse':>13}{'saved':>9}{'red%':>7}")
pf = 131072 / 5000
for bw in [0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 3.1, 12.5]:
    r = run(131072, 2048, bw, 10, 5, pf, 0.65, 0.76)
    base, sp = r["layerwise"]["ttft"], r["sparse_predicted"]["ttft"]
    xfer = (131072 * BPT) / (bw * 1e9) * 1000
    print(f"{bw:12.2f}{xfer:19.1f}{pf/L*1000:22.1f}{base*1000:15.1f}ms"
          f"{sp*1000:12.1f}ms{(base-sp)*1000:8.1f}{pct(sp,base):7.1f}%")
