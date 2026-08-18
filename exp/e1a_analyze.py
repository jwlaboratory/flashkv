"""E1a: what the traces say about the three load-bearing claims.

D2  how big is the set of KV blocks a layer must have on hand?
D3  how stable is it across decode steps / how fast does the working set grow?
D1  can the PREFILL worker predict it from the last prefill token?

Two selector architectures are evaluated, because they behave very differently:
  per_head  each head picks its own top-k  (NSA / MoBA / Quest)
            -> the KV blocks that must be resident is the UNION over heads
  shared    one top-k shared by all heads   (DeepSeek DSA lightning indexer)
            -> resident set is exactly k
"""
import glob, json, os, sys
import numpy as np

def sel_per_head(sc, k):
    """sc [H,nb] -> bool [H,nb]"""
    idx = np.argpartition(-sc, min(k, sc.shape[1]-1), axis=-1)[:, :k]
    m = np.zeros_like(sc, dtype=bool)
    np.put_along_axis(m, idx, True, axis=-1)
    return m

def sel_shared(sc, k):
    """DSA-style: one index score per (query, key) shared over heads."""
    agg = sc.sum(0)
    idx = np.argpartition(-agg, min(k, len(agg)-1))[:k]
    m = np.zeros(len(agg), dtype=bool); m[idx] = True
    return m

def jac(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0

def analyze(path, budget_fracs=(0.016, 0.0625, 0.125, 0.25), nsteps=16):
    z = np.load(path + ".npz"); meta = json.load(open(path + ".json"))
    sc = z["scores"].astype(np.float32)   # [T, L, H, nb]
    T, L, H, nb = sc.shape
    bs = meta["block_size"]
    nsteps = min(nsteps, T - 1)
    out = dict(trace=os.path.basename(path), model=meta["model"], kind=meta["kind"],
               ctx=meta["ctx"], L=L, H=H, KVH=meta["KVH"], nb=nb, block=bs, res={})
    sink, local = 1, max(1, 256 // bs)
    free = np.zeros(nb, dtype=bool); free[:sink] = True; free[-local:] = True

    for f in budget_fracs:
        k = max(1, int(round(f * nb)))
        r = dict(k=k, budget_tokens=k * bs, frac=f)
        ph_res, sh_res, ph_rec, sh_rec = [], [], [], []
        ph_jac, sh_jac = [], []
        ph_ws, sh_ws = [], []          # working set after nsteps
        ph_pred, sh_pred = [], []      # predict-from-last-prefill-token recall
        free_cov_ph, free_cov_sh = [], []
        prev_ph = prev_sh = None
        cum_ph = np.zeros((L, nb), bool); cum_sh = np.zeros((L, nb), bool)
        first_ph = first_sh = None
        pred_ph = pred_sh = None
        for t in range(T):
            for l in range(L):
                s = sc[t, l]                       # [H, nb]
                ph = sel_per_head(s, k).any(0)     # union over heads
                sh = sel_shared(s, k)
                if t == 0:                         # last prefill token = predictor
                    if pred_ph is None:
                        pred_ph = np.zeros((L, nb), bool); pred_sh = np.zeros((L, nb), bool)
                    pred_ph[l], pred_sh[l] = ph, sh
                    continue
                ph_res.append(ph.mean()); sh_res.append(sh.mean())
                tot = s.sum(-1, keepdims=True); tot[tot == 0] = 1
                ph_rec.append(float(((s * ph).sum(-1) / tot[:, 0]).mean()))
                sh_rec.append(float(((s * sh).sum(-1) / tot[:, 0]).mean()))
                free_cov_ph.append(float((ph & free).sum() / max(1, ph.sum())))
                free_cov_sh.append(float((sh & free).sum() / max(1, sh.sum())))
                if t == 1:
                    if first_ph is None:
                        first_ph = np.zeros((L, nb), bool); first_sh = np.zeros((L, nb), bool)
                    first_ph[l], first_sh[l] = ph, sh
                if prev_ph is not None and l < len(prev_ph):
                    ph_jac.append(jac(ph, prev_ph[l])); sh_jac.append(jac(sh, prev_sh[l]))
                cum_ph[l] |= ph; cum_sh[l] |= sh
            if t >= 1:
                prev_ph = [sel_per_head(sc[t, l], k).any(0) for l in range(L)]
                prev_sh = [sel_shared(sc[t, l], k) for l in range(L)]
            if t == nsteps:
                ph_ws, sh_ws = cum_ph.mean(), cum_sh.mean()
                break
        # predictability: of the blocks decode step 1 needs, what fraction did the
        # last prefill token's own selection already contain?
        r["per_head"] = dict(
            resident_frac=float(np.mean(ph_res)), mass_recall=float(np.mean(ph_rec)),
            step_jaccard=float(np.mean(ph_jac)), workingset_frac=float(ph_ws),
            pred_hit=float((first_ph & pred_ph).sum() / max(1, first_ph.sum())),
            sinklocal_share=float(np.mean(free_cov_ph)))
        r["shared"] = dict(
            resident_frac=float(np.mean(sh_res)), mass_recall=float(np.mean(sh_rec)),
            step_jaccard=float(np.mean(sh_jac)), workingset_frac=float(sh_ws),
            pred_hit=float((first_sh & pred_sh).sum() / max(1, first_sh.sum())),
            sinklocal_share=float(np.mean(free_cov_sh)))
        out["res"][f"{f:.4f}"] = r
    return out

if __name__ == "__main__":
    paths = sorted(p[:-4] for p in glob.glob("results/e1/*.npz") if "smoke" not in p)
    allr = []
    hdr = (f"{'trace':<26}{'budget':>7}{'sel':>9}{'resident':>9}{'recall':>8}"
           f"{'jaccard':>8}{'WS@16':>8}{'pred':>7}{'sink/loc':>9}")
    print(hdr); print("-" * len(hdr))
    for p in paths:
        try: r = analyze(p)
        except Exception as e: print(f"{p}: FAILED {e}"); continue
        allr.append(r)
        for f, v in r["res"].items():
            for sel in ["per_head", "shared"]:
                d = v[sel]
                print(f"{r['trace'][:25]:<26}{float(f)*100:6.1f}%{sel:>9}"
                      f"{d['resident_frac']*100:8.1f}%{d['mass_recall']:8.3f}"
                      f"{d['step_jaccard']:8.3f}{d['workingset_frac']*100:7.1f}%"
                      f"{d['pred_hit']:7.3f}{d['sinklocal_share']*100:8.1f}%")
        print()
    json.dump(allr, open("results/e1a_analysis.json", "w"), indent=1)
    print("wrote results/e1a_analysis.json")
