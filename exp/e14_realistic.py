"""E14: redo the fresh-prefill regime map with real hardware numbers.

Earlier runs assumed a prefill instance does 5,000 tok/s.  That is roughly the
per-GPU figure, not the per-INSTANCE figure -- DeepSeek's published system uses
a 4-node / 32-GPU prefill unit.  The mistake made prefill ~25x slower than
reality (26 s for 128k instead of ~1 s), which made layer-wise pipelining look
like it had unlimited headroom to hide the transfer behind.

Grounding, from DeepSeek's reported V3 inference system:
  73.7k tok/s per H800 node including a 56.3% prefix-cache hit rate
  -> ~32.2k COMPUTED tok/s per node -> ~4.0k tok/s per H800
  -> at 74 GFLOP/token (2 x 37B active) that is ~298 TFLOPS/GPU, ~30% MFU on
     fp8 H800 (989 TFLOPS dense).  Self-consistent.
"""
import json, sys
import numpy as np
sys.path.insert(0, "exp")
from e2_pipeline_sim import Link, simulate, synth_ws

L, BLOCK, BPT = 61, 64, 1152.0
KV_PER_TOKEN = L * BPT / 1024                    # KiB/token
ACTIVE_PARAMS = 37e9
TOK_S_PER_GPU = 4000.0                           # derived above

def measured_fresh():
    C = json.load(open("results/e10_longgen.json"))
    n = min(len(c["fresh_rate"]) for c in C)
    return np.mean([c["fresh_rate"][:n] for c in C], axis=0)

FRESH = measured_fresh()
print(f"DeepSeek-V3.2 KV = {KV_PER_TOKEN:.1f} KiB/token ; "
      f"prefill = {2*ACTIVE_PARAMS/1e9:.0f} GFLOP/token ; "
      f"~{TOK_S_PER_GPU:.0f} tok/s/GPU")
print()
print("=" * 100)
print("A. IS TRANSFER ACTUALLY HIDDEN BY PREFILL COMPUTE?  (the corrected question)")
print("   KV egress a prefill instance must sustain = tok/s x 70 KiB/token")
print("=" * 100)
print(f"{'prefill GPUs':>13}{'tok/s':>10}{'128k prefill':>14}{'KV egress':>12}"
      f"{'xfer @25GbE':>13}{'@100GbE':>10}{'@IB400':>9}{'  verdict (vs prefill time)'}")
for ngpu in [8, 16, 32, 64]:
    tps = ngpu * TOK_S_PER_GPU
    t_pf = 131072 / tps
    egress = tps * KV_PER_TOKEN * 1024 / 1e9
    kv = L * 131072 * BPT / 1e9
    xf = {bw: kv / bw for bw in (3.1, 12.5, 50.0)}
    worst = "transfer-bound on 25GbE" if xf[3.1] > t_pf else "hidden everywhere"
    if xf[12.5] > t_pf: worst = "transfer-bound on 25GbE + 100GbE"
    if xf[50.0] > t_pf: worst = "transfer-bound even on IB NDR"
    print(f"{ngpu:>13}{tps:>10.0f}{t_pf:>13.2f}s{egress:>10.1f}GB/s"
          f"{xf[3.1]:>12.2f}s{xf[12.5]:>9.2f}s{xf[50.0]:>8.2f}s  {worst}")

print()
print("=" * 100)
print("B. CORRECTED REGIME MAP -- TTFT reduction vs layer-wise, FRESH PREFILL")
print("   128k ctx, DSA budget, push+demand-pull, 16 decode steps")
print("=" * 100)
nb, k = 131072 // BLOCK, 2048 // BLOCK
need, pred, forced = synth_ws(L, nb, k, 16, FRESH, 0.65, 0.76, 1, 4)
BWS = [("25GbE", 3.1, 10, 5), ("100GbE", 12.5, 3, 10), ("RoCE 200Gb", 25, 2, 15),
       ("IB NDR 400Gb", 50, 1.5, 20), ("NVLink", 450, .5, 50)]
hdr = f"{'prefill instance':<28}" + "".join(f"{n:>20}" for n, *_ in BWS)
print(hdr); print("-" * len(hdr))
rows = []
for label, ngpu in [("32 GPU (real, ~1.0s)", 32), ("8 GPU (~4.1s)", 8),
                    ("128 GPU (~0.26s)", 128)]:
    t_pf = 131072 / (ngpu * TOK_S_PER_GPU)
    cells = ""
    for bn, bw, ovh, mr in BWS:
        link = Link(bw, ovh, mr)
        a = dict(rtt_us=20.0, demand=True)
        base = simulate("layerwise", need, np.zeros((L, nb), bool), L, nb,
                        BLOCK*BPT, t_pf/L, .025/L, link, **a)["ttft"]
        sp = simulate("sparse_predicted", need, pred, L, nb, BLOCK*BPT,
                      t_pf/L, .025/L, link, **a)["ttft"]
        cells += f"{(base-sp)*1000:11.0f}ms{(base-sp)/base*100:6.1f}%"
        rows.append(dict(panel="B", gpus=ngpu, link=bn, saved_ms=(base-sp)*1000,
                         pct=(base-sp)/base*100, ttft_base=base*1000))
    print(f"{label:<28}{cells}")
print("\n(earlier runs put every one of these at 0.0% -- that was the 26s prefill artifact)")
json.dump(rows, open("results/e14_realistic.json", "w"), indent=1)
print("\nwrote results/e14_realistic.json")
