"""E21: does cache-aware selection survive RETRIEVAL?  (real GPU, RULER-style)

The hardest test for a residency bias.  In needle-in-a-haystack, one specific
block holds the answer; if the selector prefers a stale-but-resident block over
the needle block even once, the answer is lost.  NLL on continuation text (§11)
cannot see this failure mode -- retrieval accuracy can.

Runs on Modal.  Prefill is dense and identical for every config, so it is done
ONCE per sample and the KV cache is cloned for each (lambda, budget) arm.
"""
import modal

img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.6.0", "transformers==4.51.3", "accelerate", "numpy"))
app = modal.App("flashkv-ruler")
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
outvol = modal.Volume.from_name("flashkv-out", create_if_missing=True)

@app.function(image=img, gpu="A100-40GB", timeout=3600,
              volumes={"/root/.cache/huggingface": hf, "/out": outvol})
def run_eval(model_id: str, ctx: int, budget_fracs: list, lams: list,
             depths: list, seeds: int, block: int, gen: int):
    import copy, json, math, os, random, re
    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    CFG = dict(mode="dense", k=32, block=block, lam=0.0)
    STATE, STATS = {}, {"fresh": 0, "sel": 0}

    def sparse_attn(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
        scaling = scaling if scaling is not None else module.head_dim ** -0.5
        ng = q.shape[1] // k.shape[1]
        ke, ve = k.repeat_interleave(ng, 1), v.repeat_interleave(ng, 1)
        w = torch.matmul(q, ke.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask[:, :, :, : ke.shape[-2]]
        w = torch.softmax(w, -1, dtype=torch.float32).to(q.dtype)
        if CFG["mode"] != "dense" and q.shape[2] == 1:
            li = module.layer_idx; B, H, _, S = w.shape
            bs = CFG["block"]; nb = (S + bs - 1) // bs; pad = nb * bs - S
            wf = torch.nn.functional.pad(w[:, :, 0, :].float(), (0, pad))
            blk = wf.view(B, H, nb, bs).sum(-1).sum(1)[0]     # shared index over heads
            kk = min(CFG["k"], nb)
            res = STATE.get(li)
            if res is None or res.shape[0] < nb:
                nr = torch.zeros(nb, dtype=torch.bool, device=w.device)
                if res is not None: nr[: res.shape[0]] = res
                res = nr; STATE[li] = res
            sc2 = blk.clone()
            if CFG["lam"] > 0:
                sc2 = sc2 + CFG["lam"] * blk.topk(kk).values.mean() * res[:nb].float()
            sel = torch.zeros(nb, dtype=torch.bool, device=w.device)
            sel[sc2.topk(kk).indices] = True
            STATS["fresh"] += int((sel & ~res[:nb]).sum()); STATS["sel"] += kk
            STATE[li][:nb] = res[:nb] | sel
            keep = sel.repeat_interleave(bs)[:S]
            w = w * keep.view(1, 1, 1, S)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
        return torch.matmul(w, ve).transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS.register("blocksparse", sparse_attn)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda").eval()

    def set_impl(name):
        model.config._attn_implementation = name
        for l in model.model.layers: l.self_attn.config._attn_implementation = name

    # ---- RULER-style haystack: repeated filler sentences, needles inserted
    FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
              "Here we go. There and back again. ")
    WORDS = ["ocean", "lantern", "compass", "meadow", "granite", "harbor",
             "willow", "cinder", "prairie", "beacon"]

    def make_sample(rng, depth, task):
        """task='multikey': 8 distractors, retrieve 1.  task='multivalue': one key
        with 4 values scattered through the haystack, retrieve ALL of them --
        four separate blocks must survive selection, which is the real stress
        test for a residency bias."""
        filler_toks = len(tok(FILLER).input_ids)
        reps = max(1, int(ctx * 1.15 / filler_toks))
        parts = [FILLER] * reps
        if task == "multikey":
            keys = rng.sample(WORDS, 8)
            vals = [rng.randint(1000000, 9999999) for _ in keys]
            for i, (kk, vv) in enumerate(zip(keys, vals)):
                d = depth if i == 0 else rng.uniform(0.05, 0.95)
                parts.insert(min(len(parts)-1, max(0, int(d*len(parts)))),
                             f"One of the special magic numbers for {kk} is: {vv}. ")
            golds = [str(vals[0])]; qk = keys[0]
            q = (f"\n\nWhat is the special magic number for {qk}? "
                 f"Answer with the number only.\nAnswer:")
        else:
            qk = rng.choice(WORDS)
            golds = [str(rng.randint(1000000, 9999999)) for _ in range(4)]
            spread = [depth * 0.5 + 0.18 * i for i in range(4)]
            for vv, d in zip(golds, spread):
                parts.insert(min(len(parts)-1, max(0, int(min(d, .95)*len(parts)))),
                             f"One of the special magic numbers for {qk} is: {vv}. ")
            for _ in range(4):                      # distractor keys
                dk = rng.choice([w for w in WORDS if w != qk])
                parts.insert(rng.randrange(len(parts)),
                             f"One of the special magic numbers for {dk} is: "
                             f"{rng.randint(1000000, 9999999)}. ")
            q = (f"\n\nList ALL of the special magic numbers for {qk}, "
                 f"comma separated.\nAnswer:")
        body = tok.decode(tok(("".join(parts)), return_tensors="pt")
                          .input_ids[0, :ctx - 80])
        msgs = [{"role": "user", "content": body + q}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt").input_ids.cuda(), golds, task

    def cache_kv(c):
        """(keys, values) per layer, across transformers cache API versions"""
        if hasattr(c, "layers"):
            return [(l.keys, l.values) for l in c.layers]
        return list(zip(c.key_cache, c.value_cache))

    def clone_cache(c):
        n = DynamicCache()
        for i, (kx, vx) in enumerate(cache_kv(c)):
            n.update(kx.clone(), vx.clone(), i)
        return n

    results = []
    rng = random.Random(0)
    samples = []
    for s in range(seeds):
        for d in depths:
            for task in ("multikey", "multivalue"):
                samples.append(make_sample(rng, d, task))
    print(f"{len(samples)} samples, ctx target {ctx}", flush=True)

    for si, (ids, golds, task) in enumerate(samples):
        S = ids.shape[1]
        set_impl("sdpa")
        base = DynamicCache()
        with torch.no_grad():
            for s0 in range(0, S - 1, 2048):
                model(ids[:, s0:min(s0 + 2048, S - 1)], past_key_values=base,
                      use_cache=True)
        nb_tot = (S + block - 1) // block
        for bf in budget_fracs:
            kk = max(1, int(round(bf * nb_tot)))
            for lam in lams:
                mode = "dense" if lam is None else "sparse"
                CFG.update(mode=mode, lam=(lam or 0.0), k=kk)
                STATE.clear(); STATS.update(fresh=0, sel=0)
                past = clone_cache(base)
                set_impl("sdpa" if mode == "dense" else "blocksparse")
                cur = ids[:, -1:]
                out = []
                with torch.no_grad():
                    for _ in range(gen):
                        o = model(cur, past_key_values=past, use_cache=True)
                        cur = o.logits[:, -1:].argmax(-1)
                        out.append(int(cur[0, 0]))
                txt = tok.decode(out)
                found = sum(1 for g in golds if g in txt)
                results.append(dict(sample=si, ctx=S, budget=bf, lam=lam, k=kk,
                                    nb=nb_tot, task=task,
                                    score=found / len(golds),
                                    correct=bool(found == len(golds)),
                                    gold=",".join(golds), out=txt[:60],
                                    fresh=STATS["fresh"] / max(1, STATS["sel"])))
                del past
        del base
        torch.cuda.empty_cache()
        if si % 3 == 0: print(f"  sample {si+1}/{len(samples)}", flush=True)
        # persist as we go so a client disconnect never loses the run
        with open("/out/e21_ruler.json", "w") as f:
            json.dump(results, f)
        outvol.commit()
    return results

@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-7B-Instruct", seqlen: int = 32768,
         seeds: int = 6, block: int = 64, gen: int = 40):
    import json
    call = run_eval.spawn(model, seqlen, [0.005, 0.0156],
                          [None, 0.0, 0.1, 0.3, 1.0],
                          [0.15, 0.35, 0.55, 0.75, 0.9], seeds, block, gen)
    print(f"spawned {call.object_id}; results stream to volume flashkv-out")
    return
    import collections
    agg = collections.defaultdict(list)
    fr = collections.defaultdict(list)
    for x in r:
        agg[(x["budget"], x["lam"])].append(x["correct"])
        fr[(x["budget"], x["lam"])].append(x["fresh"])
    print(f"\nRULER-style NIAH (4 needles), {model}, ctx~{seqlen}, n={seeds*5} per cell")
    print(f"{'budget':>9}{'selector':>22}{'accuracy':>11}{'fresh/step':>13}")
    for (b, l), v in sorted(agg.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1])):
        name = "dense" if l is None else ("block-sparse top-k" if l == 0 else f"cache-aware lam={l}")
        print(f"{b*100:8.2f}%{name:>22}{sum(v)/len(v)*100:10.1f}%"
              f"{sum(fr[(b,l)])/len(fr[(b,l)]):12.3f}")
