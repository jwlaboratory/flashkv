"""E19: cache-aware SELECTION (the co-design nobody in the prior work does).

HiSparse, SAC, InfiniGen, ArkVale, ShadowKV all take the sparse selection as
given and optimise how to fetch it.  But attention scores near the top-k cut-off
are often nearly tied -- block #33 is barely worse than block #32.  If #33 is
already resident and #32 is not, taking #33 costs almost no attention mass and
saves an entire round trip.

Rule: score' = score + lambda * (mean top-k score) for blocks already resident,
then take top-k of score'.  lambda=0 is the standard selector.

Measured: attention mass retained (quality) vs fresh blocks per step (stalls).
"""
import glob, json, os, sys
import numpy as np

def analyze(path, budget_frac=0.125, LAMBDAS=(0.0, 0.01, 0.03, 0.1, 0.3, 1.0)):
    z = np.load(path + ".npz"); meta = json.load(open(path + ".json"))
    sc = z["scores"].astype(np.float32); T, L, H, nb = sc.shape
    nt = meta.get("n_tail", 1); k = max(1, int(round(budget_frac * nb)))
    agg = sc.sum(2)
    ns = T - nt
    out = {}
    for lam in LAMBDAS:
        mass, fresh_n, ws = [], [], []
        for l in range(0, L, 2):
            resident = np.zeros(nb, bool)
            for d in range(ns):
                s = agg[nt + d, l]
                true_idx = np.argpartition(-s, min(k, nb-1))[:k]
                true_mass = s[true_idx].sum()
                if lam > 0:
                    bonus = lam * s[true_idx].mean()
                    s2 = s + bonus * resident
                else:
                    s2 = s
                pick = np.argpartition(-s2, min(k, nb-1))[:k]
                sel = np.zeros(nb, bool); sel[pick] = True
                fresh_n.append(int((sel & ~resident).sum()))
                mass.append(s[pick].sum() / max(1e-9, true_mass))
                resident |= sel
            ws.append(resident.mean())
        out[lam] = dict(mass=float(np.mean(mass)), fresh=float(np.mean(fresh_n)),
                        fresh_frac=float(np.mean(fresh_n) / k),
                        working_set=float(np.mean(ws)))
    return dict(trace=os.path.basename(path), k=k, nb=nb, res=out)

if __name__ == "__main__":
    bf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.125
    paths = sorted(p[:-4] for p in glob.glob("results/e1l/*.npz"))
    A = [analyze(p, bf) for p in paths]
    LAM = sorted(A[0]["res"])
    print(f"{len(A)} traces, 256-step generations, k={A[0]['k']}/{A[0]['nb']} blocks "
          f"({bf*100:.2f}% budget)\n")
    print(f"{'lambda':>8}{'attn mass kept':>16}{'fresh blocks/step':>19}"
          f"{'vs baseline':>13}{'working set':>13}")
    base_f = np.mean([a["res"][0.0]["fresh"] for a in A])
    rows = []
    for lam in LAM:
        m = np.mean([a["res"][lam]["mass"] for a in A])
        f = np.mean([a["res"][lam]["fresh"] for a in A])
        w = np.mean([a["res"][lam]["working_set"] for a in A])
        tag = "   <- standard selector" if lam == 0 else ""
        print(f"{lam:>8.2f}{m:15.4f} {f:>18.3f}{f/base_f:12.2f}x{w*100:12.1f}%{tag}")
        rows.append(dict(lam=lam, mass=m, fresh=f, ratio=f/base_f, ws=w))
    json.dump(rows, open(f"results/e19_cacheaware_{bf}.json", "w"), indent=1)
    print("\nattn mass kept = 1.000 means identical to the standard selector's choice")
