"""Probe MiniCPM4.1-8B (trained with InfLLM-v2 block-sparse attention) to find
where to hook the block selection."""
import modal
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.8.0", "transformers==4.57.1", "accelerate",
                    "numpy", "sentencepiece", "protobuf"))
app = modal.App("flashkv-probe")
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
outvol = modal.Volume.from_name("flashkv-out", create_if_missing=True)

@app.function(image=img, gpu="A100-80GB", timeout=2400,
              volumes={"/root/.cache/huggingface": hf, "/out": outvol})
def probe(model_id: str):
    import inspect, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    info = {"model": model_id,
            "arch": cfg.architectures,
            "layers": getattr(cfg, "num_hidden_layers", None),
            "heads": getattr(cfg, "num_attention_heads", None),
            "kv_heads": getattr(cfg, "num_key_value_heads", None),
            "max_pos": getattr(cfg, "max_position_embeddings", None),
            "sparse_cfg": {k: v for k, v in vars(cfg).items()
                           if "sparse" in k.lower() or "infllm" in k.lower()
                           or "kernel" in k.lower() or "topk" in k.lower()}}
    m = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="cuda").eval()
    attn = m.model.layers[0].self_attn
    info["attn_class"] = type(attn).__module__ + "." + type(attn).__name__
    src = inspect.getsource(type(attn).forward)
    info["uses_registry"] = "ALL_ATTENTION_FUNCTIONS" in src or "attention_interface" in src
    info["fwd_signature"] = str(inspect.signature(type(attn).forward))
    info["src_head"] = src[:1500]
    info["registry_keys"] = list(ALL_ATTENTION_FUNCTIONS.valid_keys())
    info["impl"] = m.config._attn_implementation
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    ids = tok("hello world " * 200, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        o = m(ids)
    info["forward_ok"] = bool(o.logits.shape[0])
    import json
    with open("/out/e22_probe.json", "w") as f: json.dump(info, f, default=str)
    outvol.commit()
    return info

@app.local_entrypoint()
def main(model_id: str = "openbmb/MiniCPM4.1-8B"):
    import json
    r = probe.remote(model_id)
    for k, v in r.items():
        if k == "src_head": continue
        print(f"{k}: {v}")
    print("\n--- attention forward source (head) ---")
    print(r["src_head"])
