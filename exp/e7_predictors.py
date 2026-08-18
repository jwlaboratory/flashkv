"""E7: can we predict which blocks the decode worker will need?

Ground truth: the DSA-style shared-index top-k set the decode worker actually
reads at step d, layer l.  Each predictor produces a RANKING over blocks; we
prefetch the top m*k and measure recall.  m is the over-fetch multiplier -- at a
1.56% budget, m=4 is still only 6% of the cache, so over-fetching is cheap.

Predictors, grouped by who can compute them and when:

  prefill-side (free, available before any decode happens)
    last_prefill    the last prefill token's own attention
    tail_mean/max   aggregate over the last 8 prefill positions
    sink_local      attention sinks + sliding window (query-independent)
    random          baseline

  decode-side (available during decoding, needs a fast reaction path)
    prev_step       what this layer needed at the previous decode step
    prev_layer      what the PREVIOUS LAYER needed at this same step -- usable as
                    a just-in-time prefetch hint, since layer l-1 finishes ~0.4ms
                    before layer l needs its blocks
    layer0          layer 0's true selection, used for every later layer
"""
import glob, json, os, sys
import numpy as np

def topk_mask(v, k):
    m = np.zeros(len(v), bool); m[np.argpartition(-v, min(k, len(v)-1))[:k]] = True
    return m

def recall(rank, truth, k, m):
    """fraction of `truth` inside the top m*k of `rank`"""
    n = min(int(m * k), len(rank))
    pref = np.zeros(len(rank), bool)
    pref[np.argpartition(-rank, min(n, len(rank)-1))[:n]] = True
    return float((pref & truth).sum() / max(1, truth.sum()))

def analyze(path, budget_frac=0.0156, MULT=(1, 2, 3, 4, 8)):
    z = np.load(path + ".npz"); meta = json.load(open(path + ".json"))
    sc = z["scores"].astype(np.float32)           # [T, L, H, nb]
    T, L, H, nb = sc.shape
    ntail = meta.get("n_tail", 1)
    k = max(1, int(round(budget_frac * nb)))
    agg = sc.sum(2)                                # [T, L, nb] shared index score
    dec0 = ntail                                   # first decode step index
    nsteps = T - dec0
    if nsteps < 2: return None

    sink, local = 1, max(1, 256 // meta["block_size"])
    slv = np.zeros(nb); slv[:sink] = 1e3; slv[nb-local:] = 1e3
    rng = np.random.default_rng(0)

    truth = {(d, l): topk_mask(agg[dec0 + d, l], k)
             for d in range(nsteps) for l in range(L)}
    # cross-layer agreement: do different layers want the same blocks?
    jl = []
    for d in range(min(4, nsteps)):
        for l in range(1, L):
            a, b = truth[(d, l)], truth[(d, l - 1)]
            jl.append((a & b).sum() / max(1, (a | b).sum()))

    preds = {}
    preds["last_prefill"] = lambda d, l: agg[ntail - 1, l]
    if ntail > 1:
        preds["tail_mean"] = lambda d, l: agg[:ntail, l].mean(0)
        preds["tail_max"]  = lambda d, l: agg[:ntail, l].max(0)
    preds["sink_local"] = lambda d, l: slv
    preds["random"]     = lambda d, l: rng.random(nb)
    preds["prev_step"]  = lambda d, l: agg[dec0 + d - 1, l] if d > 0 else agg[ntail - 1, l]
    preds["prev_layer"] = lambda d, l: agg[dec0 + d, l - 1] if l > 0 else agg[ntail - 1, l]
    preds["layer0"]     = lambda d, l: agg[dec0 + d, 0]

    out = dict(trace=os.path.basename(path), ctx=meta["ctx"], L=L, nb=nb, k=k,
               ntail=ntail, model=meta["model"], kind=meta["kind"],
               cross_layer_jaccard=float(np.mean(jl)), res={})
    for name, fn in preds.items():
        # first decode token only (the TTFT-critical one), and mean over all steps
        r1 = {m: np.mean([recall(fn(0, l), truth[(0, l)], k, m) for l in range(L)])
              for m in MULT}
        ra = {m: np.mean([recall(fn(d, l), truth[(d, l)], k, m)
                          for d in range(nsteps) for l in range(L)]) for m in MULT}
        out["res"][name] = dict(first_token={str(m): float(v) for m, v in r1.items()},
                                all_steps={str(m): float(v) for m, v in ra.items()})
    return out

if __name__ == "__main__":
    paths = sorted(p[:-4] for p in glob.glob("results/e1t/*.npz")) or \
            sorted(p[:-4] for p in glob.glob("results/e1/*16k_b64.npz"))
    MULT = (1, 2, 3, 4, 8)
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=float, default=0.0156)
    bf = ap.parse_args().budget
    allr = [r for r in (analyze(p, bf) for p in paths) if r]
    if not allr: sys.exit("no traces")
    print(f"sources: {', '.join(sorted({r['trace'] for r in allr}))}")
    print(f"traces: {len(allr)}  (budget k={allr[0]['k']}/{allr[0]['nb']} blocks "
          f"= {allr[0]['k']/allr[0]['nb']*100:.2f}% of cache)")
    print(f"cross-layer Jaccard (do layers agree?): "
          f"{np.mean([r['cross_layer_jaccard'] for r in allr]):.3f}")
    for which, lab in [("first_token", "FIRST DECODE TOKEN (the TTFT-critical set)"),
                       ("all_steps", "MEAN OVER ALL DECODE STEPS")]:
        print(f"\n=== recall @ over-fetch multiplier -- {lab} ===")
        print(f"{'predictor':<16}" + "".join(f"{'m='+str(m):>9}" for m in MULT) +
              f"{'% of cache at m=4':>20}")
        for name in allr[0]["res"]:
            v = [np.mean([r["res"][name][which][str(m)] for r in allr]) for m in MULT]
            f4 = allr[0]["k"] * 4 / allr[0]["nb"] * 100
            print(f"{name:<16}" + "".join(f"{x:9.3f}" for x in v) + f"{f4:19.1f}%")
    json.dump(allr, open("results/e7_predictors.json", "w"), indent=1)
    print("\nwrote results/e7_predictors.json")
