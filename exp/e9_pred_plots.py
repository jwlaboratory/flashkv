"""E9: predictability figures."""
import json
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = json.load(open("results/e7_predictors.json"))
O = json.load(open("results/e8_overfetch.json"))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
M = [1, 2, 3, 4, 8]
style = {"last_prefill": ("#2f6fb0", "-", "o", "last prefill token  (prefill-side, free)"),
         "tail_mean":    ("#5aa0d8", "-", "s", "mean of last 8 prefill tokens"),
         "prev_step":    ("#3f8f5f", "-", "^", "previous decode step  (decode-side)"),
         "prev_layer":   ("#8f7f3f", "--", "v", "previous layer, same step  (decode-side)"),
         "layer0":       ("#b08fd0", "--", "d", "layer 0's selection"),
         "sink_local":   ("#c1443c", ":", "x", "sinks + sliding window  (query-independent)"),
         "random":       ("#999999", ":", ".", "random")}
for name, (c, ls, mk, lab) in style.items():
    if name not in P[0]["res"]: continue
    y = [np.mean([r["res"][name]["first_token"][str(m)] for r in P]) for m in M]
    ax[0].plot(M, y, ls, marker=mk, color=c, lw=2, label=lab)
ax[0].axhline(1.0, color="k", lw=1, alpha=.3)
ax[0].set_xlabel("over-fetch multiplier  (x the sparse budget)")
ax[0].set_ylabel("recall of the blocks the first decode token actually reads")
ax[0].set_title("Can we predict what the decode worker will need?\n"
                "Qwen2.5-0.5B, 16k ctx, DSA-proportional budget (1.56% of cache)",
                fontsize=10)
ax[0].set_xticks(M); ax[0].set_ylim(0, 1.05); ax[0].grid(alpha=.3)
ax[0].legend(fontsize=8, loc="lower right")
ax[0].annotate("4x over-fetch = 6% of cache\n-> 94% recall", xy=(4, .938),
               xytext=(4.4, .62), fontsize=9,
               arrowprops=dict(arrowstyle="->", color="#2f6fb0"))

for ln, c in [("25GbE x-DC", "#c1443c"), ("100GbE", "#2f6fb0"), ("IB NDR 400Gb", "#3f8f5f")]:
    rs = [r for r in O if r["link"] == ln]
    ax[1].plot([r["m"] for r in rs], [r["ttft_ms"] for r in rs], "o-", color=c, lw=2, label=ln)
    ax[1].axhline(rs[0]["ttft_layerwise"], ls="--", color=c, lw=1, alpha=.6)
ax[1].set_xlabel("over-fetch multiplier"); ax[1].set_ylabel("TTFT (ms)")
ax[1].set_yscale("log")
ax[1].set_title("What that prediction is worth\n"
                "128k ctx, DeepSeek-V3.2 geometry, prefix-cache hit\n"
                "(dashed = layer-wise + demand-pull baseline)", fontsize=10)
ax[1].grid(alpha=.3); ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig("results/predictors.png", dpi=140)
print("wrote results/predictors.png")
