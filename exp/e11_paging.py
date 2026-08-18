"""E11: bulk-transfer vs remote paging, driven by the REAL 256-step traces,
plus the attack that should kill paging: round-trip latency.

A pager stalls on a round trip whenever a decode step wants a block it does not
have.  Misses inside one layer can be batched into a single request, so the cost
is ~1 RTT per layer per step in the worst case -- 61 RTTs per token for
DeepSeek-V3.2.  At 20us that is 1.2ms against a 25ms budget; at 1ms it is 61ms
and paging is dead.  Prefetching is what converts serial round trips into
pipelined ones, so this is where the prediction work earns its keep.
"""
import glob, json, sys
import numpy as np
sys.path.insert(0, "exp")
from e2_pipeline_sim import Link, simulate, synth, synth_ws

def measured_fresh():
    """mean measured fresh-block-per-step curve from the 256-step traces"""
    C = json.load(open("results/e10_longgen.json"))
    n = min(len(c["fresh_rate"]) for c in C)
    return np.mean([c["fresh_rate"][:n] for c in C], axis=0)

BPT = 1152.0

def sel_shared(sc, k):
    agg = sc.sum(0); m = np.zeros(len(agg), bool)
    m[np.argpartition(-agg, min(k, len(agg)-1))[:k]] = True
    return m

def load(path, budget_frac=0.0156, steps=256):
    z = np.load(path + ".npz"); meta = json.load(open(path + ".json"))
    sc = z["scores"].astype(np.float32); T, L, H, nb = sc.shape
    nt = meta.get("n_tail", 1); k = max(1, int(round(budget_frac * nb)))
    need = [np.stack([sel_shared(sc[t, l], k) for l in range(L)])
            for t in range(nt, min(T, nt + steps))]
    pred = np.stack([sel_shared(sc[nt - 1, l], k) for l in range(L)])
    return meta, need, pred, L, nb, k

IDXB = 128.0        # DSA lightning-indexer key: 128 dims fp8, per token per layer

POLICIES = [
    ("layer-wise bulk",        "layerwise",        "none",   True,  None, 0),
    ("sparse priority + bulk", "sparse_predicted", "pred",   True,  None, 0),
    ("paged + prefetch",       "sparse_predicted", "pred",   False, 1,    0),
    ("paged, no prefetch",     "sparse_predicted", "none",   False, 1,    0),
    ("paged, cold index",      "sparse_predicted", "pred",   False, 1,    1),
]

def run_all(need, pred, L, nb, k, block, link, rtt_us, gen):
    out = {}
    nd = need[:gen]
    for name, pol, pm, bg, cu, ix in POLICIES:
        m = pred if pm == "pred" else np.zeros((L, nb), bool)
        out[name] = simulate(pol, nd, m, L, nb, block * BPT, 0.0, .025 / L, link,
                             rtt_us=rtt_us, demand=True, background=bg, cold_unit=cu,
                             index_bytes=ix * nb * block * IDXB)
    return out

print("=" * 104)
print("A. REAL 256-STEP TRACES, prefix-cache hit, 100GbE, 20us RTT")
print("   (trace geometry: Qwen 24 layers -- absolute ms are small, the RATIOS are the point)")
print("=" * 104)
paths = sorted(p[:-4] for p in glob.glob("results/e1l/*.npz"))
link = Link(12.5, 3, 10)
print(f"{'trace':<14}{'gen':>5}{'policy':<24}{'TTFT ms':>9}{'TPOT ms':>9}"
      f"{'GB moved':>10}{'vs bulk':>9}")
rows = []
for p in paths[:2]:
    meta, need, pred, L, nb, k = load(p)
    for gen in [16, 64, 256]:
        r = run_all(need, pred, L, nb, k, meta["block_size"], link, 20.0, gen)
        base = r["layer-wise bulk"]["total_bytes"]
        for name in r:
            d = r[name]
            print(f"{meta['kind'][:13]:<14}{gen:>5}{name:<24}{d['ttft']*1000:9.2f}"
                  f"{d['tpot']*1000:9.2f}{d['total_bytes']/1e9:10.3f}"
                  f"{d['total_bytes']/base:8.2f}x")
            rows.append(dict(trace=meta["kind"], gen=gen, policy=name,
                             ttft=d["ttft"]*1000, tpot=d["tpot"]*1000,
                             gb=d["total_bytes"]/1e9))
        print()

print("=" * 104)
print("B. THE ATTACK: round-trip latency.  DeepSeek geometry (61 layers), 128k ctx,")
print("   100GbE, 64 decode steps.  A pager pays up to 1 RTT per layer per token.")
print("=" * 104)
L, BLOCK = 61, 64
nb, k = 131072 // BLOCK, 2048 // BLOCK
FRESH = measured_fresh()
need, pred, _ = synth_ws(L, nb, k, 64, FRESH, 0.65, 0.76, 1, 4)
print(f"{'RTT':>8}{'61xRTT/token':>14}" +
      "".join(f"{n:>26}" for n, *_ in POLICIES[:1] + POLICIES[2:4]))
print(f"{'':>22}" + "".join(f"{'TTFT / TPOT ms':>26}" for _ in range(3)))
for rtt in [10, 20, 50, 100, 200, 500, 1000]:
    r = run_all(need, pred, L, nb, k, BLOCK, Link(12.5, 3, 10), float(rtt), 64)
    c = ""
    for name, *_ in POLICIES[:1] + POLICIES[2:4]:
        d = r[name]
        c += f"{d['ttft']*1000:12.1f} /{d['tpot']*1000:7.1f}   "
    print(f"{rtt:6d}us{L*rtt/1000:12.1f}ms{c}")
    rows.append(dict(rtt_us=rtt, **{n: r[n]["tpot"]*1000 for n, *_ in POLICIES}))
json.dump(rows, open("results/e11_paging.json", "w"), indent=1)
print("\nideal TPOT with zero transfer cost = 25.0 ms")
print("wrote results/e11_paging.json")


print("=" * 104)
print("C. THE INDEX CACHE IS THE PAGER'S CRITICAL PATH")
print("   A DSA pager cannot pick a block until it holds the lightning-indexer keys for")
print("   EVERY token (128 B/tok/layer fp8 = 11% of the KV, 1.02 GB at 128k).")
print("   256 decode steps, 128k ctx, 20us RTT.")
print("=" * 104)
print(f"{'link':<15}{'policy':<26}{'TTFT ms':>9}{'TPOT ms':>9}{'GB moved':>10}{'% of KV':>9}")
need, pred, _ = synth_ws(L, nb, k, 256, FRESH, 0.65, 0.76, 1, 4)
print(f"   calibrated synthetic working set @256 = "
      f"{np.stack(need).any(0).mean()*100:.1f}% (measured: 15.2%)\n")
full = L * nb * BLOCK * BPT
for lname, bw, ovh, mr in [("IB NDR 400Gb", 50, 1.5, 20), ("100GbE", 12.5, 3, 10),
                           ("25GbE x-DC", 3.1, 10, 5)]:
    r = run_all(need, pred, L, nb, k, BLOCK, Link(bw, ovh, mr), 20.0, 256)
    for name in [n for n, *_ in POLICIES if n != "paged, no prefetch"]:
        d = r[name]
        print(f"{lname:<15}{name:<26}{d['ttft']*1000:9.1f}{d['tpot']*1000:9.1f}"
              f"{d['total_bytes']/1e9:10.2f}{d['total_bytes']/full*100:8.1f}%")
        rows.append(dict(panel="C", link=lname, policy=name, ttft=d["ttft"]*1000,
                         tpot=d["tpot"]*1000, gb=d["total_bytes"]/1e9))
    print()
json.dump(rows, open("results/e11_paging.json", "w"), indent=1)
