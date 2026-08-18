"""E23: cache-aware selection on a model TRAINED for sparse attention.

MiniCPM4.1-8B is trained with InfLLM-v2, a trainable block-sparse attention
(NSA-family): each token attends to <5% of a 128K context by design.  That makes
it the right subject -- Qwen2.5 (§12) was never trained to tolerate sparsity, so
its attention distributions are not adapted to it.

Harder and closer to the real setting than §12:
  - 64k context (the model's native window) rather than 32k
  - k=32 selected blocks, DeepSeek DSA's actual block count
  - a tight k=8 arm to probe the threshold found in §12
  - three RULER tasks including multi-value (4 blocks must all survive) and
    multi-query (two independent needles in one question)

MiniCPM defines its own attention classes rather than using the HF registry, so
the block-sparse selector is installed by binding a replacement forward to each
layer; prefill delegates to the model's own fast path untouched.
"""
import modal

img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate",
                    "numpy", "sentencepiece", "protobuf"))
app = modal.App("flashkv-minicpm")
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
outvol = modal.Volume.from_name("flashkv-out", create_if_missing=True)

@app.function(image=img, gpu="A100-80GB", timeout=7200,
              volumes={"/root/.cache/huggingface": hf, "/out": outvol})
def run(model_id: str, seqlen: int, ks: list, lams: list, bonus_modes: list,
        depths: list, seeds: int, block: int, gen: int, tag: str):
    import json, math, random, re, sys, types
    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", device_map="cuda").eval()
    mod = sys.modules[type(model.model.layers[0].self_attn).__module__]
    apply_rope, repeat_kv = mod.apply_rotary_pos_emb, mod.repeat_kv

    CFG = dict(mode="dense", k=32, block=block, lam=0.0, bonus="mean")
    STATE, STATS = {}, {"fresh": 0, "sel": 0}

    def sparse_forward(self, hidden_states, attention_mask=None, position_ids=None,
                       past_key_value=None, output_attentions=False,
                       use_cache=False, **kw):
        bsz, q_len, _ = hidden_states.size()
        if CFG["mode"] == "dense" or q_len != 1:
            return self._orig_forward(hidden_states, attention_mask=attention_mask,
                                      position_ids=position_ids,
                                      past_key_value=past_key_value,
                                      output_attentions=output_attentions,
                                      use_cache=use_cache, **kw)
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads,
                                            self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads,
                                            self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads,
                                            self.head_dim).transpose(1, 2)
        kv_len = position_ids.max().item() + 1
        cos, sin = self.rotary_emb(v.to(torch.float32), seq_len=kv_len)
        q, k = apply_rope(q, k, cos, sin, position_ids)
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx,
                                         {"sin": sin, "cos": cos})
        ke = repeat_kv(k, self.num_key_value_groups)
        ve = repeat_kv(v, self.num_key_value_groups)
        w = torch.matmul(q, ke.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            w = w + attention_mask[:, :, :, : ke.shape[-2]]
        w = torch.softmax(w, -1, dtype=torch.float32).to(q.dtype)

        li = self.layer_idx
        B, H, _, S = w.shape
        bs = CFG["block"]; nb = (S + bs - 1) // bs; pad = nb * bs - S
        wf = torch.nn.functional.pad(w[:, :, 0, :].float(), (0, pad))
        blk = wf.view(B, H, nb, bs).sum(-1).sum(1)[0]     # index shared over heads
        kk = min(CFG["k"], nb)
        res = STATE.get(li)
        if res is None or res.shape[0] < nb:
            nr = torch.zeros(nb, dtype=torch.bool, device=w.device)
            if res is not None: nr[: res.shape[0]] = res
            res = nr; STATE[li] = res
        sc = blk.clone()
        if CFG["lam"] > 0:
            tv = blk.topk(kk).values
            ref = tv.mean() if CFG["bonus"] == "mean" else tv.min()
            sc = sc + CFG["lam"] * ref * res[:nb].float()
        sel = torch.zeros(nb, dtype=torch.bool, device=w.device)
        sel[sc.topk(kk).indices] = True
        STATS["fresh"] += int((sel & ~res[:nb]).sum()); STATS["sel"] += kk
        STATE[li][:nb] = res[:nb] | sel
        keep = sel.repeat_interleave(bs)[:S]
        w = w * keep.view(1, 1, 1, S)
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
        o = torch.matmul(w, ve).transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(o), None, past_key_value

    for lyr in model.model.layers:
        a = lyr.self_attn
        a._orig_forward = a.forward
        a.forward = types.MethodType(sparse_forward, a)

    FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
              "Here we go. There and back again. ")
    WORDS = ["ocean", "lantern", "compass", "meadow", "granite", "harbor",
             "willow", "cinder", "prairie", "beacon", "quarry", "thistle"]

    def make(rng, depth, task):
        ftok = len(tok(FILLER).input_ids)
        parts = [FILLER] * max(1, int(seqlen * 1.1 / ftok))
        def put(d, s):
            parts.insert(min(len(parts)-1, max(0, int(min(d, .96)*len(parts)))), s)
        if task == "multikey":
            keys = rng.sample(WORDS, 8)
            vals = [rng.randint(1000000, 9999999) for _ in keys]
            for i, (kx, vv) in enumerate(zip(keys, vals)):
                put(depth if i == 0 else rng.uniform(.05, .95),
                    f"One of the special magic numbers for {kx} is: {vv}. ")
            golds = [str(vals[0])]
            q = (f"\n\nWhat is the special magic number for {keys[0]}? "
                 f"Answer with the number only.\nAnswer:")
        elif task == "multivalue":
            qk = rng.choice(WORDS)
            golds = [str(rng.randint(1000000, 9999999)) for _ in range(4)]
            for i, vv in enumerate(golds):
                put(depth * .5 + .18 * i,
                    f"One of the special magic numbers for {qk} is: {vv}. ")
            for _ in range(4):
                dk = rng.choice([w for w in WORDS if w != qk])
                put(rng.uniform(.05, .95), f"One of the special magic numbers "
                    f"for {dk} is: {rng.randint(1000000,9999999)}. ")
            q = (f"\n\nList ALL of the special magic numbers for {qk}, "
                 f"comma separated.\nAnswer:")
        else:                                   # multiquery: two needles at once
            k1, k2 = rng.sample(WORDS, 2)
            golds = [str(rng.randint(1000000, 9999999)) for _ in range(2)]
            put(depth, f"One of the special magic numbers for {k1} is: {golds[0]}. ")
            put(min(.95, depth + .4),
                f"One of the special magic numbers for {k2} is: {golds[1]}. ")
            for _ in range(6):
                dk = rng.choice([w for w in WORDS if w not in (k1, k2)])
                put(rng.uniform(.05, .95), f"One of the special magic numbers "
                    f"for {dk} is: {rng.randint(1000000,9999999)}. ")
            q = (f"\n\nWhat are the special magic numbers for {k1} and {k2}? "
                 f"Answer with the two numbers only.\nAnswer:")
        body = tok.decode(tok("".join(parts), return_tensors="pt")
                          .input_ids[0, :seqlen - 90])
        text = tok.apply_chat_template([{"role": "user", "content": body + q}],
                                       tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt").input_ids.cuda(), golds, task

    def cache_kv(c):
        if hasattr(c, "layers"): return [(l.keys, l.values) for l in c.layers]
        return list(zip(c.key_cache, c.value_cache))

    rng = random.Random(0)
    samples = [make(rng, d, t) for _ in range(seeds) for d in depths
               for t in ("multikey", "multivalue", "multiquery")]
    print(f"{len(samples)} samples, target seqlen {seqlen}", flush=True)
    # --- sanity gate: dense attention must retrieve, else the harness is broken
    ids0, golds0, _ = samples[0]
    S0 = ids0.shape[1]
    CFG["mode"] = "dense"
    c0 = DynamicCache()
    with torch.no_grad():
        for s0 in range(0, S0 - 1, 4096):
            e0 = min(s0 + 4096, S0 - 1)
            model(ids0[:, s0:e0],
                  position_ids=torch.arange(s0, e0, device=ids0.device).unsqueeze(0),
                  past_key_values=c0, use_cache=True)
        cur, o2 = ids0[:, -1:], []
        for t in range(gen):
            r0 = model(cur, position_ids=torch.tensor([[S0-1+t]], device=ids0.device),
                       past_key_values=c0, use_cache=True)
            cur = r0.logits[:, -1:].argmax(-1); o2.append(int(cur[0, 0]))
    probe_txt = tok.decode(o2)
    ok = any(g in re.sub(r"[,\s]", "", probe_txt) for g in golds0)
    print(f"SANITY dense@{S0}: retrieved={ok} out={probe_txt[:80]!r}", flush=True)
    with open(f"/out/e23{tag}_sanity.json", "w") as f:
        json.dump({"seqlen": S0, "ok": bool(ok), "out": probe_txt[:200],
                   "gold": golds0}, f)
    outvol.commit()
    del c0; torch.cuda.empty_cache()
    if not ok:
        return {"sanity_failed": True, "seqlen": S0, "out": probe_txt[:200]}
    results = []
    for si, (ids, golds, task) in enumerate(samples):
        S = ids.shape[1]
        CFG["mode"] = "dense"
        base = DynamicCache()
        with torch.no_grad():
            for s0 in range(0, S - 1, 4096):
                e0 = min(s0 + 4096, S - 1)
                pos = torch.arange(s0, e0, device=ids.device).unsqueeze(0)
                model(ids[:, s0:e0], position_ids=pos, past_key_values=base,
                      use_cache=True)
        nb_tot = (S + block - 1) // block
        for kk in ks:
            for bm in bonus_modes:
                for lam in lams:
                    if lam == 0.0 and bm != bonus_modes[0]: continue
                    if lam is None and (kk != ks[0] or bm != bonus_modes[0]): continue
                    CFG.update(mode=("dense" if lam is None else "sparse"),
                               lam=(lam or 0.0), k=kk, bonus=bm)
                    STATE.clear(); STATS.update(fresh=0, sel=0)
                    past = DynamicCache()
                    for i, (kx, vx) in enumerate(cache_kv(base)):
                        past.update(kx.clone(), vx.clone(), i)
                    cur = ids[:, -1:]; outs = []
                    with torch.no_grad():
                        for t in range(gen):
                            pos = torch.tensor([[S - 1 + t]], device=ids.device)
                            o = model(cur, position_ids=pos,
                                      past_key_values=past, use_cache=True)
                            cur = o.logits[:, -1:].argmax(-1)
                            outs.append(int(cur[0, 0]))
                    txt = tok.decode(outs)
                    flat = re.sub(r"[,\s]", "", txt)
                    found = sum(1 for g in golds if g in flat)
                    results.append(dict(sample=si, ctx=S, k=kk, nb=nb_tot, lam=lam,
                                        bonus=bm, task=task, score=found/len(golds),
                                        correct=bool(found == len(golds)),
                                        out=txt[:60], gold=",".join(golds),
                                        fresh=STATS["fresh"]/max(1, STATS["sel"])))
                    del past
        del base
        torch.cuda.empty_cache()
        with open(f"/out/e23{tag}.json", "w") as f: json.dump(results, f)
        outvol.commit()
        print(f"  sample {si+1}/{len(samples)} ({task})", flush=True)
    return results
