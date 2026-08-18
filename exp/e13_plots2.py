"""E13: figures for the paging round."""
import json, sys
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "exp")
from e2_pipeline_sim import Link, simulate, synth_ws

C = json.load(open("results/e10_longgen.json"))
E = json.load(open("results/e11_paging.json"))
fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))

# (1) measured working-set saturation
for c in C:
    ax[0,0].plot(range(1, len(c["curve"])+1), np.array(c["curve"])*100, lw=2,
                 label=f"{c['trace'].replace('q05_','Qwen0.5B ').replace('q15_','Qwen1.5B ')} "
                       f"({c['ctx']//1024}k)")
ax[0,0].axhline(100, color="k", ls="--", lw=1, alpha=.4)
ax[0,0].text(130, 88, "what a bulk transfer sends", fontsize=9, alpha=.7)
ax[0,0].set_xlabel("decode steps"); ax[0,0].set_ylabel("cumulative KV touched (%)")
ax[0,0].set_ylim(0, 105)
ax[0,0].set_title("(1) The working set SATURATES\n"
                  "256 tokens of generation touch ~15% of the cache;\n"
                  "fresh blocks/step falls to 0.1-1.2%", fontsize=10)
ax[0,0].legend(fontsize=8, loc="center right"); ax[0,0].grid(alpha=.3)

# (2) policy head-to-head at 100GbE
P = [r for r in E if r.get("panel") == "C" and r["link"] == "100GbE"]
names = [r["policy"] for r in P]
short = [n.replace("layer-wise ", "layer-wise\n").replace("sparse priority + ", "sparse prio\n+ ")
          .replace("paged + ", "paged\n+ ").replace("paged, ", "paged\n") for n in names]
x = np.arange(len(P))
a2 = ax[0,1]; a3 = a2.twinx()
b = a2.bar(x-0.2, [r["ttft"] for r in P], 0.4, color="#2f6fb0", label="TTFT (ms)")
a2.bar_label(b, fmt="%.0f", fontsize=8)
b2 = a3.bar(x+0.2, [r["gb"] for r in P], 0.4, color="#c1443c", label="GB moved")
a3.bar_label(b2, fmt="%.2f", fontsize=8)
a2.set_xticks(x); a2.set_xticklabels(short, fontsize=8)
a2.set_ylabel("TTFT (ms)", color="#2f6fb0"); a3.set_ylabel("GB moved", color="#c1443c")
a2.set_title("(2) 128k ctx, 100GbE, 256-token generation\n"
             "paging moves 15% of the KV -- but only if the\n"
             "indexer keys are already resident", fontsize=10)

# (3) RTT: where paging breaks
R = [r for r in E if "rtt_us" in r]
for key, c, lab in [("layer-wise bulk", "#888888", "layer-wise bulk"),
                    ("sparse priority + bulk", "#2f6fb0", "sparse priority + bulk"),
                    ("paged + prefetch", "#3f8f5f", "paged + prefetch")]:
    ax[1,0].plot([r["rtt_us"] for r in R], [r[key] for r in R], "o-", color=c, lw=2, label=lab)
ax[1,0].axhline(25, color="k", ls=":", lw=1); ax[1,0].text(11, 25.7, "ideal TPOT", fontsize=8)
ax[1,0].axvline(100, color="#c1443c", ls="--", lw=1.5)
ax[1,0].text(115, 40, "paging loses\nabove ~100us RTT", fontsize=9, color="#c1443c")
ax[1,0].set_xscale("log"); ax[1,0].set_yscale("log")
ax[1,0].set_xlabel("round-trip latency to the KV store (us)")
ax[1,0].set_ylabel("TPOT (ms/token)")
ax[1,0].set_title("(3) The attack that bounds paging\n"
                  "a pager pays up to 1 RTT per layer per token (61x)", fontsize=10)
ax[1,0].legend(fontsize=8); ax[1,0].grid(alpha=.3)

# (4) capacity
kv, ws = 9.21, 0.152
HBM, W = 80.0, 40.0
labels = ["bulk\n(full KV)", "paged\n(15% + index)"]
gb = [kv, kv * (ws + 0.11)]
hbm_batch = [(HBM - W) / g for g in gb]
bw_rps = [[b / g for g in gb] for b in [12.5, 50.0]]
xx = np.arange(2)
b1 = ax[1,1].bar(xx-0.27, hbm_batch, 0.25, color="#8f5fb0", label="HBM-capped batch (80GB H100)")
b2 = ax[1,1].bar(xx, [r*20 for r in bw_rps[0]], 0.25, color="#2f6fb0",
                 label="concurrent reqs, 100GbE")
b3 = ax[1,1].bar(xx+0.27, [r*20 for r in bw_rps[1]], 0.25, color="#3f8f5f",
                 label="concurrent reqs, IB NDR")
for bb in (b1, b2, b3): ax[1,1].bar_label(bb, fmt="%.0f", fontsize=8)
ax[1,1].set_xticks(xx); ax[1,1].set_xticklabels(labels, fontsize=9)
ax[1,1].set_ylabel("concurrent 128k requests")
ax[1,1].set_title("(4) Capacity, not just latency\n"
                  "bulk forces the whole KV resident; paging needs\n"
                  "only the working set + index", fontsize=10)
ax[1,1].legend(fontsize=8); ax[1,1].grid(alpha=.3, axis="y")

plt.tight_layout(); plt.savefig("results/paging.png", dpi=140)
print("wrote results/paging.png")
