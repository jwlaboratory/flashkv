"""E20: does cache-aware selection actually cost model quality?

E19 measured retained attention MASS, which is a proxy.  This runs the real
thing: a custom attention implementation that does DSA-style block-sparse
selection with a residency bonus, and teacher-forces a held-out continuation to
measure negative log-likelihood per token.  Compared against dense attention and
against the standard (lambda=0) top-k selector.

  lambda=0    standard block-sparse selector
  lambda>0    prefer blocks already resident in the decode worker's cache when
              their score is within lambda*(mean top-k score) of the cut-off
"""
import argparse, json, math, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

CFG = dict(mode="dense", k=32, block=64, lam=0.0)
STATE = {}      # layer -> resident bool tensor
STATS = {"fresh": 0, "sel": 0, "steps": 0}

def sparse_attn(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
    scaling = scaling if scaling is not None else module.head_dim ** -0.5
    ng = q.shape[1] // k.shape[1]
    ke = k.repeat_interleave(ng, 1); ve = v.repeat_interleave(ng, 1)
    w = torch.matmul(q, ke.transpose(2, 3)) * scaling
    if attention_mask is not None:
        w = w + attention_mask[:, :, :, : ke.shape[-2]]
    w = torch.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)

    if CFG["mode"] != "dense" and q.shape[2] == 1:          # decode step only
        li = module.layer_idx
        B, H, _, S = w.shape
        bs = CFG["block"]; nb = (S + bs - 1) // bs
        pad = nb * bs - S
        wf = torch.nn.functional.pad(w[:, :, 0, :].float(), (0, pad))
        blk = wf.view(B, H, nb, bs).sum(-1).sum(1)[0]        # shared index: sum over heads
        kk = min(CFG["k"], nb)
        res = STATE.get(li)
        if res is None or res.shape[0] < nb:
            nr = torch.zeros(nb, dtype=torch.bool, device=w.device)
            if res is not None: nr[: res.shape[0]] = res
            res = nr; STATE[li] = res
        score = blk.clone()
        if CFG["lam"] > 0:
            topv = blk.topk(kk).values.mean()
            score = score + CFG["lam"] * topv * res[:nb].float()
        pick = score.topk(kk).indices
        sel = torch.zeros(nb, dtype=torch.bool, device=w.device)
        sel[pick] = True
        STATS["fresh"] += int((sel & ~res[:nb]).sum()); STATS["sel"] += kk
        STATE[li][:nb] = res[:nb] | sel
        keep = sel.repeat_interleave(bs)[:S]
        w = w * keep.view(1, 1, 1, S)
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
    return torch.matmul(w, ve).transpose(1, 2).contiguous(), None

ALL_ATTENTION_FUNCTIONS.register("blocksparse", sparse_attn)

def set_impl(model, name):
    model.config._attn_implementation = name
    for lyr in model.model.layers:
        lyr.self_attn.config._attn_implementation = name

def run(model, ids, ctx, n_eval, mode, lam, k, block):
    CFG.update(mode=mode, lam=lam, k=k, block=block)
    STATE.clear(); STATS.update(fresh=0, sel=0, steps=0)
    past = DynamicCache()
    with torch.no_grad():
        set_impl(model, "sdpa")            # prefill is identical for all modes
        for s in range(0, ctx, 1024):
            model(ids[:, s:min(s + 1024, ctx)], past_key_values=past, use_cache=True)
        set_impl(model, "blocksparse")
        nll = []
        for t in range(n_eval):
            o = model(ids[:, ctx + t - 1: ctx + t], past_key_values=past, use_cache=True)
            lp = torch.log_softmax(o.logits[0, -1].float(), -1)
            nll.append(-lp[ids[0, ctx + t]].item())
    return np.array(nll), STATS["fresh"] / max(1, STATS["sel"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--eval", type=int, default=128)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--budget", type=float, default=0.125)
    ap.add_argument("--windows", type=int, default=3)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, attn_implementation="blocksparse").to(dev).eval()
    txt = open("data/tale.txt", encoding="utf-8", errors="ignore").read()
    txt = re.sub(r"\n{3,}", "\n\n", txt[txt.find("A TALE OF TWO CITIES"):])
    nb = (a.ctx + a.eval) // a.block
    k = max(1, int(round(a.budget * nb)))
    print(f"{a.model}  ctx={a.ctx}  eval={a.eval} tokens/window x {a.windows} windows  "
          f"block={a.block}  k={k}/{nb} ({a.budget*100:.1f}%)\n")
    LAMS = [0.0, 0.03, 0.1, 0.3, 1.0]
    acc = {"dense": []}; fr_acc = {}
    for lam in LAMS: acc[lam] = []
    full = tok(txt, return_tensors="pt").input_ids
    for w in range(a.windows):
        off = w * (a.ctx + a.eval + 64)
        ids = full[:, off: off + a.ctx + a.eval + 8].to(dev)
        if ids.shape[1] < a.ctx + a.eval + 1: break
        d, _ = run(model, ids, a.ctx, a.eval, "dense", 0.0, k, a.block)
        acc["dense"].append(d)
        for lam in LAMS:
            n, fr = run(model, ids, a.ctx, a.eval, "sparse", lam, k, a.block)
            acc[lam].append(n); fr_acc.setdefault(lam, []).append(fr)
        print(f"  window {w+1}/{a.windows} done", flush=True)
    D = np.concatenate(acc["dense"])
    T0 = np.concatenate(acc[0.0])
    n = len(D)
    print(f"\n{n} paired eval tokens. Differences are PAIRED (same tokens, same context);")
    print("+-  is a 95% CI on the paired mean difference.\n")
    print(f"{'selector':<26}{'NLL':>8}{'vs dense':>18}{'vs top-k':>20}{'fresh':>9}{'ratio':>8}")
    def ci(x): return 1.96 * x.std(ddof=1) / math.sqrt(len(x))
    print(f"{'dense (no sparsity)':<26}{D.mean():8.4f}{'--':>18}{'--':>20}{'--':>9}{'--':>8}")
    rows = [dict(sel="dense", nll=float(D.mean()))]
    for lam in LAMS:
        X = np.concatenate(acc[lam]); fr = float(np.mean(fr_acc[lam]))
        dd, dt = X - D, X - T0
        nm = "block-sparse top-k" if lam == 0 else f"cache-aware lam={lam}"
        sig = "" if lam == 0 else ("  *" if abs(dt.mean()) > ci(dt) else "  ns")
        print(f"{nm:<26}{X.mean():8.4f}{dd.mean():+11.4f}+-{ci(dd):.4f}"
              f"{dt.mean():+13.4f}+-{ci(dt):.4f}{fr:9.3f}"
              f"{fr/np.mean(fr_acc[0.0]):7.2f}x{sig}")
        rows.append(dict(sel=nm, lam=lam, nll=float(X.mean()),
                         vs_dense=float(dd.mean()), vs_topk=float(dt.mean()),
                         ci_vs_topk=float(ci(dt)), fresh=fr,
                         ratio=float(fr/np.mean(fr_acc[0.0]))))
    json.dump(rows, open("results/e20_quality.json", "w"), indent=1)
    print("\n* = statistically significant vs the standard top-k selector; ns = not")
