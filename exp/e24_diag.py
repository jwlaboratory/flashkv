"""Isolate: is it chunked prefill, the reasoning template, or long context?"""
import modal
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate",
                    "numpy", "sentencepiece", "protobuf"))
app = modal.App("flashkv-diag")
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
outvol = modal.Volume.from_name("flashkv-out", create_if_missing=True)

@app.function(image=img, gpu="A100-80GB", timeout=3600,
              volumes={"/root/.cache/huggingface": hf, "/out": outvol})
def diag(model_id: str):
    import json, re, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", device_map="cuda").eval()
    # A single repeated sentence is a pathological haystack -- the model
    # degenerates into copying it.  Real RULER uses varied natural prose.
    import urllib.request
    try:
        raw = urllib.request.urlopen(
            "https://www.gutenberg.org/files/98/98-0.txt", timeout=60
        ).read().decode("utf-8", "ignore")
        NAT = re.sub(r"\n{2,}", "\n", raw[raw.find("A TALE OF TWO CITIES"):])
    except Exception:
        NAT = None
    FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
              "Here we go. There and back again. ")
    GOLD = "7429183"
    out = []
    for seqlen in [16384, 32768, 65536]:
      for hay in (["repeat", "natural"] if NAT else ["repeat"]):
        if hay == "repeat":
            ft = len(tok(FILLER).input_ids)
            parts = [FILLER] * max(1, int(seqlen * 1.05 / ft))
        else:
            ids_n = tok(NAT[: seqlen * 8], return_tensors="pt").input_ids[0]
            chunk = max(1, len(ids_n) // 200)
            parts = [tok.decode(ids_n[i:i+chunk]) for i in range(0, len(ids_n), chunk)]
        parts.insert(len(parts)//2,
                     f" One of the special magic numbers for ocean is: {GOLD}. ")
        body = tok.decode(tok("".join(parts), return_tensors="pt").input_ids[0, :seqlen-80])
        q = ("\n\nWhat is the special magic number for ocean? "
             "Answer with the number only.\nAnswer:")
        for tmpl in ["chat_nothink"]:
            if tmpl == "raw":
                text = body + q
            else:
                kw = {} if tmpl == "chat" else {"enable_thinking": False}
                try:
                    text = tok.apply_chat_template(
                        [{"role": "user", "content": body + q}], tokenize=False,
                        add_generation_prompt=True, **kw)
                except TypeError:
                    continue
            ids = tok(text, return_tensors="pt").input_ids.cuda()
            S = ids.shape[1]
            for mode in ["single"]:
                try:
                    c = DynamicCache()
                    with torch.no_grad():
                        if mode == "single":
                            m(ids[:, :S-1], past_key_values=c, use_cache=True)
                        else:
                            for s0 in range(0, S-1, 4096):
                                e0 = min(s0+4096, S-1)
                                m(ids[:, s0:e0],
                                  position_ids=torch.arange(s0, e0, device="cuda").unsqueeze(0),
                                  past_key_values=c, use_cache=True)
                        cur, o = ids[:, -1:], []
                        for t in range(48):
                            r = m(cur, past_key_values=c, use_cache=True)
                            cur = r.logits[:, -1:].argmax(-1); o.append(int(cur[0,0]))
                    txt = tok.decode(o)
                    hit = GOLD in re.sub(r"[,\s]", "", txt)
                except Exception as e:
                    txt, hit = f"ERR {type(e).__name__}: {e}"[:120], False
                out.append(dict(seqlen=S, tmpl=tmpl, prefill=mode, hay=hay,
                                hit=bool(hit), out=txt[:110]))
                print(f"{S:>7} {hay:<8} hit={hit}  {txt[:70]!r}", flush=True)
                del c; torch.cuda.empty_cache()
    with open("/out/e24_diag2.json", "w") as f: json.dump(out, f)
    outvol.commit()
    return out
