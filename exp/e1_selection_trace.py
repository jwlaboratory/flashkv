"""E1: oracle block-selection traces from real LLMs.

At each decode step we capture the EXACT per-head attention distribution over the
whole KV cache, bucket it into blocks, and ask what a block-sparse selector
(DSA / NSA / MoBA / Quest style) would have to fetch.

Oracle selection by true attention mass is the *upper bound* on sparsity and
stability: a trained selector can only be worse. So if the oracle's critical set
is large or unstable, the priority-transfer idea is dead.
"""
import argparse, json, os, re, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- prompts
def load_corpus():
    with open("data/tale.txt", encoding="utf-8", errors="ignore") as f:
        t = f.read()
    t = t[t.find("A TALE OF TWO CITIES"):]
    return re.sub(r"\n{3,}", "\n\n", t)

def load_code():
    import torch as _t
    p = os.path.join(os.path.dirname(_t.__file__), "nn", "modules", "module.py")
    return open(p, encoding="utf-8", errors="ignore").read()

def build_prompt(kind, tok, target_tokens):
    """Return prompt text sized to ~target_tokens."""
    if kind == "code":
        body = load_code()
    else:
        body = load_corpus()
    # binary-search a character budget
    lo, hi = 100, len(body)
    while lo < hi - 200:
        mid = (lo + hi) // 2
        n = len(tok(body[:mid]).input_ids)
        if n < target_tokens - 120:
            lo = mid
        else:
            hi = mid
    body = body[:lo]
    if kind == "qa_needle":
        # plant a fact ~40% in, ask about it at the end
        cut = int(len(body) * 0.4)
        needle = "\n\nImportant record: the vault combination at Tellson's Bank is 7741-Q.\n\n"
        body = body[:cut] + needle + body[cut:]
        return body + "\n\nQuestion: What is the vault combination at Tellson's Bank?\nAnswer:"
    if kind == "summarize":
        return body + "\n\nWrite a detailed summary of the passage above.\nSummary:"
    if kind == "code":
        return body + "\n\n# Question: explain what the _apply method does.\n# Answer:"
    return body  # 'continue'

# ---------------------------------------------------------------- capture
class AttnCapture:
    """Grabs attn_weights from each self_attn module output (eager impl only)."""
    def __init__(self, model):
        self.store, self.on, self.handles = {}, False, []
        for i, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.self_attn.register_forward_hook(self._mk(i)))
    def _mk(self, idx):
        def hook(mod, inp, out):
            if self.on and isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                self.store[idx] = out[1].detach().float().cpu()  # [B,H,q,S]
        return hook
    def clear(self): self.store = {}
    def close(self):
        for h in self.handles: h.remove()

# ---------------------------------------------------------------- selection
def block_scores(attn_1xS, block_size):
    """attn_1xS: [H, S] float32 attention probs for one query position.
    returns [H, nblocks] summed mass."""
    H, S = attn_1xS.shape
    nb = (S + block_size - 1) // block_size
    pad = nb * block_size - S
    if pad:
        attn_1xS = torch.nn.functional.pad(attn_1xS, (0, pad))
    return attn_1xS.view(H, nb, block_size).sum(-1)

def topk_sets(scores, k):
    """scores [H,nb] -> bool [H,nb] top-k per head."""
    H, nb = scores.shape
    k = min(k, nb)
    idx = scores.topk(k, dim=-1).indices
    m = torch.zeros(H, nb, dtype=torch.bool)
    m.scatter_(1, idx, True)
    return m

def mass_sets(scores, tau):
    """minimal blocks per head covering tau of the mass."""
    H, nb = scores.shape
    srt, idx = scores.sort(dim=-1, descending=True)
    cum = srt.cumsum(-1)
    # number needed = first index where cum >= tau, +1
    need = (cum < tau).sum(-1) + 1
    m = torch.zeros(H, nb, dtype=torch.bool)
    for h in range(H):
        m[h, idx[h, :need[h]]] = True
    return m

def forced_sets(nb, S, block_size, n_sink_blocks, local_tokens):
    """Blocks a scheme keeps unconditionally: attention sinks + sliding window.
    P can predict these with ZERO knowledge of the decode query."""
    m = torch.zeros(nb, dtype=torch.bool)
    m[:n_sink_blocks] = True
    first_local = max(0, (S - local_tokens)) // block_size
    m[first_local:] = True
    return m

# ---------------------------------------------------------------- main
def run(model_id, kind, ctx, steps, block_size, budget_tokens, tau,
        n_sink_blocks, local_tokens, out):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    KVH = getattr(cfg, "num_key_value_heads", H)
    grp = H // KVH
    print(f"{model_id}: L={L} H={H} KVH={KVH} group={grp}", flush=True)

    text = build_prompt(kind, tok, ctx)
    ids = tok(text, return_tensors="pt").input_ids[:, :ctx].to(dev)
    S0 = ids.shape[1]
    print(f"prompt={kind} tokens={S0}", flush=True)

    cap = AttnCapture(model)
    t0 = time.time()
    # --- chunked prefill of all but the last token, with fast sdpa
    from transformers import DynamicCache
    past = DynamicCache()
    CH = 1024
    with torch.no_grad():
        for s in range(0, S0 - 1, CH):
            chunk = ids[:, s:min(s + CH, S0 - 1)]
            out_ = model(chunk, past_key_values=past, use_cache=True)
            past = out_.past_key_values
    print(f"prefill {time.time()-t0:.1f}s", flush=True)

    # --- single-token steps with eager so we get attention weights
    model.config._attn_implementation = "eager"
    for lyr in model.model.layers:
        lyr.self_attn.config._attn_implementation = "eager"
    cap.on = True

    records = []          # list of dicts, one per (step)
    sel_union = []        # [steps][L] bool arrays over blocks (MLA-style shared KV)
    sel_kv = []           # [steps][L] count of (kvhead, block) pairs needed
    cur = ids[:, -1:]
    with torch.no_grad():
        for t in range(steps + 1):     # t=0 is the LAST PREFILL TOKEN
            cap.clear()
            o = model(cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            S = past.layers[0].keys.shape[2] if hasattr(past, "layers") else o.past_key_values[0][0].shape[2]
            nb = (S + block_size - 1) // block_size
            k = max(1, budget_tokens // block_size)
            step_union, step_kv, per_layer = [], [], []
            for l in range(L):
                a = cap.store[l][0, :, -1, :]        # [H, S]
                sc = block_scores(a, block_size)     # [H, nb]
                tk = topk_sets(sc, k)
                ms = mass_sets(sc, tau)
                forced = forced_sets(nb, S, block_size, n_sink_blocks, local_tokens)
                # recall of top-k, and of forced-only (sink+local, query-independent)
                rec_tk = (sc * tk).sum(-1).mean().item()
                rec_forced = (sc * forced.unsqueeze(0)).sum(-1).mean().item()
                u_tk = tk.any(0)                     # shared-KV (MLA) union over heads
                u_ms = ms.any(0)
                # GQA: block needed for kv-head g if any q head in its group picks it
                tk_g = tk.view(KVH, grp, nb).any(1)  # [KVH, nb]
                ms_g = ms.view(KVH, grp, nb).any(1)
                per_layer.append(dict(
                    nb=nb,
                    topk_perhead=int(tk.sum(-1).float().mean().item()),
                    topk_union=int(u_tk.sum()),
                    mass_perhead=float(ms.sum(-1).float().mean().item()),
                    mass_union=int(u_ms.sum()),
                    gqa_topk_frac=float(tk_g.float().mean().item()),
                    gqa_mass_frac=float(ms_g.float().mean().item()),
                    recall_topk=rec_tk,
                    recall_sinklocal=rec_forced,
                    forced_frac=float(forced.float().mean().item()),
                ))
                step_union.append(u_ms.numpy())
                step_kv.append(u_tk.numpy())
            records.append(dict(step=t, S=S, nb=nb, layers=per_layer))
            sel_union.append(step_union)
            sel_kv.append(step_kv)
            cur = o.logits[:, -1:].argmax(-1)
            if t % 4 == 0: print(f"  step {t} S={S} nb={nb}", flush=True)
    cap.close()

    nb_max = max(r["nb"] for r in records)
    def pack(sel):
        arr = np.zeros((len(sel), L, nb_max), dtype=bool)
        for i, st in enumerate(sel):
            for l, v in enumerate(st):
                arr[i, l, :len(v)] = v
        return arr
    meta = dict(model=model_id, kind=kind, ctx=S0, steps=steps, L=L, H=H, KVH=KVH,
                block_size=block_size, budget_tokens=budget_tokens, tau=tau,
                n_sink_blocks=n_sink_blocks, local_tokens=local_tokens,
                records=records)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out + ".npz", sel_mass=pack(sel_union), sel_topk=pack(sel_kv))
    json.dump(meta, open(out + ".json", "w"))
    print("wrote", out, flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--kind", default="qa_needle")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--budget", type=int, default=1024)
    p.add_argument("--tau", type=float, default=0.95)
    p.add_argument("--sink-blocks", type=int, default=1)
    p.add_argument("--local", type=int, default=256)
    p.add_argument("--out", default="results/e1/run")
    a = p.parse_args()
    run(a.model, a.kind, a.ctx, a.steps, a.block, a.budget, a.tau,
        a.sink_blocks, a.local, a.out)
