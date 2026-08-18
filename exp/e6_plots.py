"""E6: figures for the four decisive measurements."""
import glob, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "exp")
from e1a_analyze import sel_per_head, sel_shared
from e2_pipeline_sim import Link, simulate, synth

C = dict(ph="#c1443c", sh="#2f6fb0", lw="#888888", ok="#3f8f5f")
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# ---- (1) resident set vs budget: per-head union blows up, shared does not
A = json.load(open("results/e1a_analysis.json"))
main = [r for r in A if "16k_b64" in r["trace"]]
fr = sorted(float(f) for f in main[0]["res"])
for sel, c, lab in [("per_head", C["ph"], "per-head top-k (NSA/MoBA/Quest)\n-> union over heads"),
                    ("shared", C["sh"], "shared index (DeepSeek DSA)\n-> exactly the budget")]:
    Y = np.array([[r["res"][f"{f:.4f}"][sel]["resident_frac"] for f in fr] for r in main])
    ax[0,0].plot(np.array(fr)*100, Y.mean(0)*100, "o-", color=c, lw=2, label=lab)
    ax[0,0].fill_between(np.array(fr)*100, Y.min(0)*100, Y.max(0)*100, color=c, alpha=.15)
ax[0,0].plot([0,25],[0,25],"--",color="k",lw=1,alpha=.4)
ax[0,0].set_xlabel("per-head attention budget (% of context)")
ax[0,0].set_ylabel("KV cache that must be resident (%)")
ax[0,0].set_title("(1) D2: does head-union destroy sparsity?\n"
                  "6 traces, Qwen2.5-0.5B/1.5B, 16k ctx, 64-tok blocks", fontsize=10)
ax[0,0].legend(fontsize=8); ax[0,0].grid(alpha=.3)

# ---- (2) working-set growth across decode steps
for path, style in [("results/e1/q05_qa_needle_16k_b64", "-"),
                    ("results/e1/q05_qa_32k_b64", "--")]:
    z = np.load(path+".npz"); m = json.load(open(path+".json"))
    sc = z["scores"].astype(np.float32); T,L,H,nb = sc.shape
    for f, c in [(0.016, C["sh"]), (0.0625, C["ok"])]:
        k = max(1, int(round(f*nb)))
        for sel, cc, lb in [(sel_shared, c, "shared"), 
                            (lambda s,k: sel_per_head(s,k).any(0), C["ph"], "per-head")]:
            if lb=="per-head" and f!=0.0625: continue
            cum = np.zeros((L,nb),bool); ys=[]
            for t in range(1, min(T,17)):
                for l in range(L): cum[l] |= sel(sc[t,l],k)
                ys.append(cum.mean()*100)
            ax[0,1].plot(range(1,len(ys)+1), ys, style, color=cc, lw=2,
                         label=f"{lb} {f*100:.1f}% budget, {m['ctx']//1024}k ctx")
ax[0,1].set_xlabel("decode steps"); ax[0,1].set_ylabel("cumulative KV touched (%)")
ax[0,1].set_title("(2) D3: how fast does the working set grow?\n"
                  "how much can stay 'lazy'", fontsize=10)
ax[0,1].legend(fontsize=7); ax[0,1].grid(alpha=.3)

# ---- (3) regime map
L61, BLOCK, BPT = 61, 64, 1152.
def ttft(pf, bw, ovh, mr, pol, demand=True, jac=.65, ph=.76):
    nb, k = 131072//BLOCK, 2048//BLOCK
    need, pred, forced = synth(L61, nb, k, 8, jac, ph, 1, 4)
    m = dict(layerwise=np.zeros((L61,nb),bool), sparse_oracle=need[0],
             sparse_predicted=pred)[pol]
    return simulate(pol, need, m, L61, nb, BLOCK*BPT, pf/L61, .025/L61,
                    Link(bw,ovh,mr), rtt_us=20., demand=demand)["ttft"]
BWS=[("25GbE\nx-DC",3.1,10,5),("100GbE",12.5,3,10),("RoCE\n200Gb",25,2,15),
     ("IB NDR\n400Gb",50,1.5,20),("NVLink",450,.5,50)]
PFS=[("prefix-cache hit\n(no prefill)",0.),("prefill 20k tok/s",131072/20000),
     ("prefill 5k tok/s",131072/5000),("prefill 1k tok/s",131072/1000)]
M=np.zeros((len(PFS),len(BWS)))
for i,(pn,pf) in enumerate(PFS):
    for j,(bn,bw,o,mr) in enumerate(BWS):
        b=ttft(pf,bw,o,mr,"layerwise"); s=ttft(pf,bw,o,mr,"sparse_predicted")
        M[i,j]=(b-s)/b*100
im=ax[1,0].imshow(M,cmap="RdYlGn",vmin=0,vmax=70,aspect="auto")
ax[1,0].set_xticks(range(len(BWS))); ax[1,0].set_xticklabels([b[0] for b in BWS],fontsize=8)
ax[1,0].set_yticks(range(len(PFS))); ax[1,0].set_yticklabels([p[0] for p in PFS],fontsize=8)
for i in range(len(PFS)):
    for j in range(len(BWS)):
        ax[1,0].text(j,i,f"{M[i,j]:.0f}%",ha="center",va="center",
                     fontsize=10,fontweight="bold")
ax[1,0].set_title("(3) D5: TTFT reduction vs layer-wise\n"
                  "the idea only exists in the top row", fontsize=10)
plt.colorbar(im,ax=ax[1,0],label="% TTFT saved")

# ---- (4) push-only vs push+pull
labels=["layer-wise","sparse\npredicted","sparse\noracle"]
x=np.arange(3); w=.35
for off,(mode,dm,c) in enumerate([("push-only",False,C["lw"]),("push + demand-pull",True,C["sh"])]):
    v=[ttft(0.,12.5,3,10,p,demand=dm)*1000 for p in
       ["layerwise","sparse_predicted","sparse_oracle"]]
    b=ax[1,1].bar(x+off*w-w/2,v,w,label=mode,color=c)
    ax[1,1].bar_label(b,fmt="%.0f",fontsize=8)
ax[1,1].set_xticks(x); ax[1,1].set_xticklabels(labels,fontsize=9)
ax[1,1].set_ylabel("TTFT (ms)"); ax[1,1].set_yscale("log")
ax[1,1].set_title("(4) D1: a push-only sparse schedule is worthless\n"
                  "128k ctx, 100GbE, prefix-cache hit", fontsize=10)
ax[1,1].legend(fontsize=8); ax[1,1].grid(alpha=.3,axis="y")

plt.tight_layout(); plt.savefig("results/findings.png",dpi=140)
print("wrote results/findings.png")
