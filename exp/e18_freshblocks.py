"""E18: what predicts the FRESH blocks?

A resident cache (LRU, HiSparse-style) already holds everything recently used.
So the only blocks that ever cause a stall are the ones entering a layer's
selection for the FIRST time.  Forwarding the current selection is useless
against these by construction -- its overlap with the next step is exactly the
part already in cache.  So: is anything else predictive of the fresh set?

Candidate predictors, all cheap and available at step t-1:
  rank      blocks ranked just below the cut-off (k+1..k+P) -- what a 2k-slot
            cache buys, i.e. HiSparse's design
  spatial   blocks NEIGHBOURING the current selection.  Untested anywhere I
            could find, and plausible: attention is often locally coherent, so a
            block entering the top-k may sit beside one already in it.
  recent    blocks that were fresh at the previous step (momentum)
  random    baseline
"""
import glob, json, os
import numpy as np

def topk_idx(v, n):
    return np.argpartition(-v, min(n, len(v)-1))[:n]

def analyze(path, budget_frac=0.0156, BUD=(8, 16, 32, 64), DIST=2):
    z = np.load(path + ".npz"); meta = json.load(open(path + ".json"))
    sc = z["scores"].astype(np.float32); T, L, H, nb = sc.shape
    nt = meta.get("n_tail", 1); k = max(1, int(round(budget_frac * nb)))
    agg = sc.sum(2)
    ns = T - nt
    PRED = ["rank", "spatial", "rank+spatial", "random"]
    res = {b: {p: [] for p in PRED} for b in BUD}
    fresh_counts, sel_counts = [], []
    rng = np.random.default_rng(0)
    for l in range(0, L, 2):                       # subsample layers for speed
        resident = np.zeros(nb, bool)
        prev_sel = None; prev_fresh = np.zeros(nb, bool)
        for d in range(ns):
            sel = np.zeros(nb, bool); sel[topk_idx(agg[nt + d, l], k)] = True
            fresh = sel & ~resident
            if prev_sel is not None and fresh.sum() > 0:
                fresh_counts.append(fresh.sum()); sel_counts.append(k)
                order = np.argsort(-agg[nt + d - 1, l])   # rank at step t-1
                # spatial: neighbours of the previous selection
                nb_mask = np.zeros(nb, bool)
                pi = np.flatnonzero(prev_sel)
                for off in range(1, DIST + 1):
                    nb_mask[np.clip(pi - off, 0, nb-1)] = True
                    nb_mask[np.clip(pi + off, 0, nb-1)] = True
                nb_mask &= ~prev_sel & ~resident
                for B in BUD:
                    # rank: highest-scoring non-resident blocks below the cut
                    cand = [i for i in order if not resident[i] and not prev_sel[i]][:B]
                    rk = np.zeros(nb, bool); rk[cand] = True
                    # spatial: nearest neighbours first, capped at B
                    spi = np.flatnonzero(nb_mask)
                    sp = np.zeros(nb, bool)
                    if len(spi): sp[spi[:B]] = True
                    # hybrid: half budget each
                    hy = np.zeros(nb, bool)
                    hy[cand[:B//2]] = True
                    if len(spi): hy[spi[:B - B//2]] = True
                    rd = np.zeros(nb, bool)
                    free = np.flatnonzero(~resident & ~prev_sel)
                    if len(free): rd[rng.choice(free, min(B, len(free)), replace=False)] = True
                    f = max(1, fresh.sum())
                    res[B]["rank"].append((rk & fresh).sum() / f)
                    res[B]["spatial"].append((sp & fresh).sum() / f)
                    res[B]["rank+spatial"].append((hy & fresh).sum() / f)
                    res[B]["random"].append((rd & fresh).sum() / f)
            resident |= sel; prev_sel = sel; prev_fresh = fresh
    return dict(trace=os.path.basename(path), k=k, nb=nb, L=L,
                fresh_per_step=float(np.mean(fresh_counts)),
                fresh_frac=float(np.mean(fresh_counts) / k),
                res={b: {p: float(np.mean(v)) for p, v in d.items()}
                     for b, d in res.items()})

if __name__ == "__main__":
    paths = sorted(p[:-4] for p in glob.glob("results/e1l/*.npz"))
    A = [analyze(p) for p in paths]
    print(f"{len(A)} traces, 256-step generations, k={A[0]['k']} blocks/layer "
          f"of {A[0]['nb']}\n")
    print(f"fresh blocks per layer per step: {np.mean([a['fresh_per_step'] for a in A]):.2f} "
          f"({np.mean([a['fresh_frac'] for a in A])*100:.1f}% of the selection)")
    print("\nfraction of FRESH blocks covered, by prefetch budget (blocks/layer/step):")
    print(f"{'predictor':<16}" + "".join(f"{'B='+str(b):>10}" for b in [8, 16, 32, 64]))
    for p in ["rank", "spatial", "rank+spatial", "random"]:
        print(f"{p:<16}" + "".join(
            f"{np.mean([a['res'][b][p] for a in A]):10.3f}" for b in [8, 16, 32, 64]))
    json.dump(A, open("results/e18_fresh.json", "w"), indent=1)
    print("\n(B=32 equals one extra selection's worth of traffic per layer per step)")
    print("wrote results/e18_fresh.json")
