"""E12: is there a real window at fleet scale, or does it close?

A bulk transfer pays for the whole KV once.  A pager pays for whatever the
generation touches, spread over the whole generation -- so paging must lose
eventually, on a long enough generation.  Where is the crossover, and what do
the two policies cost in link bandwidth and decode-worker HBM at realistic
concurrency?

HBM matters as much as bandwidth: a bulk transfer forces the decode worker to
hold the ENTIRE KV resident, a pager only the working set.  That sets how many
concurrent long-context requests fit on a GPU at all.
"""
import json, os
import numpy as np

L, BPT, BLOCK = 61, 1152.0, 64          # DeepSeek-V3.2 MLA geometry
AMP = 1.0                                # pager fetches exactly the block it needs
LINKS = [("25GbE x-DC", 3.1), ("100GbE", 12.5), ("RoCE 200Gb", 25.0),
         ("IB NDR 400Gb", 50.0), ("PCIe4 host", 24.0)]
HBM_GB = 80.0                            # H100
WEIGHTS_GB = 40.0                        # per-GPU share of a big MoE, illustrative

def ws_curve():
    """measured cumulative-working-set curves from E10, if present"""
    p = "results/e10_longgen.json"
    if not os.path.exists(p): return None
    return json.load(open(p))

def ws_at(curves, steps):
    """mean measured working-set fraction after `steps` decode steps"""
    v = []
    for c in curves:
        cu = c["curve"]
        v.append(cu[min(steps, len(cu)) - 1])
    return float(np.mean(v))

C = ws_curve()
if not C:
    raise SystemExit("run exp/e10_longgen.py first")
print(f"measured working-set curves from {len(C)} traces "
      f"({', '.join(str(c['ctx']) for c in C)} ctx)\n")

for ctx in [32768, 131072]:
    kv = L * ctx * BPT / 1e9
    print("=" * 96)
    print(f"### ctx={ctx}  full KV = {kv:.2f} GB/request")
    print("=" * 96)
    print(f"{'gen len':>8}{'working set':>13}{'bulk GB/req':>13}{'pager GB/req':>14}"
          f"{'pager/bulk':>12}   verdict")
    for g in [16, 32, 64, 128, 256, 512, 1024]:
        w = ws_at(C, g)
        if g > max(len(c["curve"]) for c in C):
            # extrapolate: fit the tail slope of the measured curve (blocks/step)
            base = ws_at(C, 256)
            slope = (ws_at(C, 256) - ws_at(C, 192)) / 64
            w = min(1.0, base + slope * (g - 256))
            note = " (extrapolated)"
        else:
            note = ""
        pager = kv * w * AMP
        v = "pager wins" if pager < kv else "BULK WINS"
        print(f"{g:>8}{w*100:12.1f}%{kv:13.2f}{pager:14.2f}{pager/kv:11.2f}x   {v}{note}")
    print()

    print(f"{'link':<15}{'policy':<10}{'GB/req':>9}{'max req/s':>11}"
          f"{'max concurrent':>16}{'HBM/req GB':>12}{'HBM-capped batch':>18}")
    for g in [128, 512]:
        w = ws_at(C, min(g, 256))
        if g > 256:
            slope = (ws_at(C, 256) - ws_at(C, 192)) / 64
            w = min(1.0, ws_at(C, 256) + slope * (g - 256))
        gen_s = g / 40.0                       # 40 tok/s decode
        for lname, bw in LINKS:
            for pol, gb, hbm in [("bulk", kv, kv), ("pager", kv * w * AMP, kv * w)]:
                rps = bw / gb
                conc = rps * gen_s
                batch = (HBM_GB - WEIGHTS_GB) / hbm
                print(f"{lname:<15}{pol:<10}{gb:9.2f}{rps:11.2f}{conc:16.1f}"
                      f"{hbm:12.2f}{batch:18.1f}")
            print()
        print(f"   ^ generation length {g} tokens, working set {w*100:.1f}%\n")
        break
