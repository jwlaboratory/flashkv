"""E2 (v2): does sparse-priority ordering actually buy anything?

Simulates the P->D link as a serial, non-preemptive, priority-scheduled channel
and walks the decode worker layer by layer.  A block that has not arrived when
the decode worker reaches it triggers a DEMAND FETCH (promoted to head of queue)
and the decode worker stalls.

Two trace sources:
  --trace   real E1 traces (Qwen 16k geometry)
  --synth   synthetic traces at DeepSeek-V3.2 geometry (61 layers, 128k ctx,
            MLA latent 576 dims bf16), calibrated with the Jaccard stability and
            prefill-predictability measured in E1.

Two scenarios, and they give opposite answers:
  fresh prefill            there is prefill compute to hide the transfer behind
  prefix-cache hit         KV is loaded from a remote store / CPU / SSD; there
                           is NO compute to hide behind
"""
import argparse, json, math, os
import numpy as np

class Link:
    def __init__(self, peak_gbps, ovh_us, msg_rate_mops, max_msg=1 << 20):
        self.peak, self.ovh = peak_gbps * 1e9, ovh_us * 1e-6
        self.min_t, self.max_msg = 1e-6 / msg_rate_mops, max_msg
    def time(self, nbytes):
        nmsg = max(1, math.ceil(nbytes / self.max_msg))
        return max(self.ovh * nmsg + nbytes / self.peak, self.min_t * nmsg)

LINKS = {                       # peak GB/s, per-msg us, Mmsg/s
    "NVLink 900Gb/s":  (450.0, 0.5, 50.0),
    "IB NDR 400Gb":    (50.0, 1.5, 20.0),
    "RoCE 200Gb":      (25.0, 2.0, 15.0),
    "100GbE":          (12.5, 3.0, 10.0),
    "25GbE cross-DC":  (3.1, 10.0, 5.0),
    "PCIe4 host KV":   (24.0, 2.0, 15.0),
}

MAX_UNIT_BLOCKS = 16      # bound a single transfer unit (~1MB) so that a demand
                          # fetch never has to drag a whole layer behind it

def runs(mask, cap=MAX_UNIT_BLOCKS):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]: j += 1
            for s in range(i, j, cap):
                out.append((s, min(cap, j - s)))
            i = j
        else: i += 1
    return out

def simulate(policy, need, prio, L, nb, block_bytes, t_pf_layer, t_dec_layer,
             link, rtt_us=0.0, index_frac=0.11, demand=True):
    """need[step][layer] bool[nb]; prio[layer] bool[nb] = what P sends first.

    demand=True   receiver may pull a missing block out of order, costing one
                  RTT (push + demand-pull, what you would actually build)
    demand=False  pure push: the receiver just waits for the stream to reach the
                  block in the sender's chosen order
    Event-driven: `future` holds units by release time, `ready` is a priority
    heap; demand fetches are located per layer by bisect and pulled out lazily."""
    import bisect, heapq, itertools
    units = []
    for l in range(L):
        rel = (l + 1) * t_pf_layer
        if policy in ("blocking", "layerwise", "sparse_pull"):
            if policy == "blocking":     r = L * t_pf_layer
            elif policy == "layerwise":  r = rel
            else:  # index cache must land, then a request round-trip
                r = rel + index_frac * nb * block_bytes / link.peak + rtt_us * 1e-6
            for s0 in range(0, nb, MAX_UNIT_BLOCKS):
                units.append((l, s0, min(MAX_UNIT_BLOCKS, nb - s0), 0, r))
        else:
            for s0, c in runs(prio[l]):   units.append((l, s0, c, l, rel))
            for s0, c in runs(~prio[l]):  units.append((l, s0, c, L + l, rel))
    U = [dict(layer=a, start=b, count=c, prio=d, release=e,
              bytes=c * block_bytes, done=False) for a, b, c, d, e in units]

    by_layer = {l: [] for l in range(L)}
    for i, u in enumerate(U): by_layer[u["layer"]].append(i)
    for l in by_layer: by_layer[l].sort(key=lambda i: U[i]["start"])
    starts = {l: [U[i]["start"] for i in by_layer[l]] for l in range(L)}

    future = [(u["release"], i) for i, u in enumerate(U)]
    heapq.heapify(future)
    ready = []
    arrived = np.zeros((L, nb), bool)
    link_t = dec_t = stall = stall_post = moved = 0.0
    ttft = bytes_ttft = None

    def release_upto(t):
        while future and future[0][0] <= t:
            _, i = heapq.heappop(future)
            if not U[i]["done"]:
                heapq.heappush(ready, (U[i]["prio"], U[i]["layer"], U[i]["start"], i))

    def run_unit(i, rtt=0.0):
        nonlocal link_t, moved
        u = U[i]
        link_t = max(link_t, u["release"]) + rtt + link.time(u["bytes"])
        arrived[u["layer"], u["start"]:u["start"] + u["count"]] = True
        u["done"] = True; moved += u["bytes"]
        return link_t

    def pop_ready():
        while ready:
            _, _, _, i = heapq.heappop(ready)
            if not U[i]["done"]: return i
        return None

    def find_covering(l, blk):
        """pending unit in layer l covering block blk"""
        j = bisect.bisect_right(starts[l], blk) - 1
        while j >= 0:
            i = by_layer[l][j]
            u = U[i]
            if u["start"] + u["count"] > blk:
                if not u["done"]: return i
            if u["start"] + MAX_UNIT_BLOCKS < blk - MAX_UNIT_BLOCKS: break
            j -= 1
        return None

    n_pending = len(U)
    for step in range(len(need)):
        for l in range(L):
            want = need[step][l][:nb]
            while True:
                missing = np.flatnonzero(want & ~arrived[l])
                if not len(missing): break
                if demand:                       # pull out of order, pay one RTT
                    i = find_covering(l, int(missing[0]))
                    rtt = rtt_us * 1e-6
                else:                            # pure push: wait for the stream
                    release_upto(max(link_t, dec_t))
                    i = pop_ready()
                    if i is None and future:
                        link_t = max(link_t, future[0][0]); continue
                    rtt = 0.0
                if i is None: break
                t_arr = run_unit(i, rtt); n_pending -= 1
                if t_arr > dec_t:
                    d = t_arr - dec_t; stall += d
                    if step > 0: stall_post += d
                    dec_t = t_arr
            dec_t += t_dec_layer
            while n_pending:                      # link works during compute
                release_upto(max(link_t, dec_t))
                i = pop_ready()
                if i is None:
                    if not future: break
                    link_t = max(link_t, future[0][0]); continue
                if max(link_t, U[i]["release"]) + link.time(U[i]["bytes"]) > dec_t:
                    heapq.heappush(ready, (U[i]["prio"], U[i]["layer"],
                                           U[i]["start"], i)); break
                run_unit(i); n_pending -= 1
        if step == 0: ttft, bytes_ttft = dec_t, moved
    while n_pending:                              # drain
        release_upto(link_t)
        i = pop_ready()
        if i is None:
            if not future: break
            link_t = max(link_t, future[0][0]); continue
        run_unit(i); n_pending -= 1
    ideal = len(need) * L * t_dec_layer
    return dict(policy=policy, ttft=ttft, total=dec_t, stall=stall,
                stall_post_ttft=stall_post, decode_ideal=ideal,
                bytes_before_ttft=bytes_ttft, link_done=link_t)

# ------------------------------------------------------- synthetic trace
def synth(L, nb, k, steps, jaccard, pred_hit, sink, local, seed=0):
    rng = np.random.default_rng(seed)
    forced = np.zeros(nb, bool); forced[:sink] = True; forced[nb - local:] = True
    keep = 2 * jaccard / (1 + jaccard)
    need, cur = [], []
    for l in range(L):
        m = forced.copy()
        free = np.flatnonzero(~forced)
        extra = max(0, k - forced.sum())
        m[rng.choice(free, min(extra, len(free)), replace=False)] = True
        cur.append(m)
    pred = []
    for l in range(L):
        p = cur[l].copy()
        on = np.flatnonzero(p & ~forced)
        drop = rng.choice(on, int(round((1 - pred_hit) * len(on))), replace=False)
        p[drop] = False
        off = np.flatnonzero(~p)
        p[rng.choice(off, min(len(drop), len(off)), replace=False)] = True
        pred.append(p)
    for t in range(steps):
        need.append(np.stack(cur))
        nxt = []
        for l in range(L):
            m = forced.copy()
            on = np.flatnonzero(cur[l] & ~forced)
            nkeep = int(round(keep * len(on)))
            m[rng.choice(on, nkeep, replace=False)] = True
            off = np.flatnonzero(~m)
            add = max(0, k - m.sum())
            m[rng.choice(off, min(add, len(off)), replace=False)] = True
            nxt.append(m)
        cur = nxt
    return need, np.stack(pred), forced

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synth", "trace"], default="synth")
    ap.add_argument("--trace", default="results/e1/q05_qa_needle_16k_b64")
    ap.add_argument("--L", type=int, default=61)          # DeepSeek-V3.2
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--bpt", type=float, default=1152.0)  # MLA 576 dims bf16
    ap.add_argument("--budget", type=int, default=2048)   # DSA top-2048 tokens
    ap.add_argument("--jaccard", type=float, default=0.65)
    ap.add_argument("--pred-hit", type=float, default=0.76)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--dec-ms", type=float, default=25.0)
    ap.add_argument("--pf-tok-s", type=float, default=5000.0)
    ap.add_argument("--rtt-us", type=float, default=20.0)
    ap.add_argument("--out", default="results/e2_sim.json")
    a = ap.parse_args()

    if a.mode == "synth":
        L, nb = a.L, a.ctx // a.block
        k = a.budget // a.block
        need, pred, forced = synth(L, nb, k, a.steps, a.jaccard, a.pred_hit,
                                   1, max(1, 256 // a.block))
        ctx = a.ctx
    else:
        z = np.load(a.trace + ".npz"); meta = json.load(open(a.trace + ".json"))
        sc = z["scores"].astype(np.float32); T, L, H, nb = sc.shape
        k = max(1, a.budget // meta["block_size"])
        def sh(s):
            agg = s.sum(0); m = np.zeros(nb, bool)
            m[np.argpartition(-agg, min(k, nb - 1))[:k]] = True; return m
        need = [np.stack([sh(sc[t, l]) for l in range(L)])
                for t in range(1, min(T, a.steps + 1))]
        pred = np.stack([sh(sc[0, l]) for l in range(L)])
        forced = np.zeros(nb, bool); forced[0] = True
        forced[nb - max(1, 256 // meta["block_size"]):] = True
        ctx = meta["ctx"]; a.block = meta["block_size"]

    block_bytes = a.block * a.bpt
    total = L * nb * block_bytes
    t_dec_layer = a.dec_ms / 1000.0 / L
    masks = dict(sparse_oracle=need[0], sparse_predicted=pred,
                 sparse_sinklocal=np.tile(forced, (L, 1)))
    print(f"geometry L={L} nb={nb} ctx={ctx} KV={total/1e9:.2f} GB "
          f"({total/ctx/1024:.0f} KB/token) budget={k}/{nb} blocks "
          f"({k/nb*100:.2f}%)")
    for kk, v in masks.items():
        print(f"  priority set {kk:<18}{v.mean()*100:6.2f}% of cache")
    print(f"  need overlap oracle vs predicted: "
          f"{(need[0] & pred).sum()/need[0].sum():.3f}")
    ws = np.zeros_like(need[0])
    for n in need: ws |= n
    print(f"  working set after {len(need)} steps: {ws.mean()*100:.2f}% of cache")

    POL = ["blocking", "layerwise", "sparse_oracle", "sparse_predicted",
           "sparse_sinklocal", "sparse_pull"]
    rows = []
    for scen, pf in [("fresh prefill", ctx / a.pf_tok_s), ("prefix-cache hit", 0.0)]:
        for ln, (bw, ovh, mr) in LINKS.items():
            link = Link(bw, ovh, mr)
            for p in POL:
                pm = masks.get(p, np.zeros((L, nb), bool))
                r = simulate(p, need, pm, L, nb, block_bytes, pf / L,
                             t_dec_layer, link, a.rtt_us)
                r.update(scenario=scen, link=ln, prefill_s=pf)
                rows.append(r)
    json.dump(dict(meta=vars(a), L=L, nb=nb, total_bytes=total, rows=rows),
              open(a.out, "w"), indent=1)

    for scen in ["fresh prefill", "prefix-cache hit"]:
        print(f"\n### {scen}: TTFT ms above prefill compute  (x = vs layerwise)")
        print(f"{'link':<17}" + "".join(f"{p.replace('sparse_','sp_'):>18}" for p in POL))
        for ln in LINKS:
            sub = {r["policy"]: r for r in rows
                   if r["scenario"] == scen and r["link"] == ln}
            base = sub["layerwise"]["ttft"] - sub["layerwise"]["prefill_s"]
            c = ""
            for p in POL:
                v = (sub[p]["ttft"] - sub[p]["prefill_s"]) * 1000
                c += f"{v:10.1f}({v/(base*1000):5.2f}x)" if base > 0 else f"{v:10.1f}(  -  )"
            print(f"{ln:<17}{c}")
        print(f"{'abs TTFT ms':<17}" + "".join(
            f"{np.mean([r['ttft'] for r in rows if r['scenario']==scen and r['policy']==p])*1000:17.1f} "
            for p in POL))
        print(f"{'post-TTFT stall':<17}" + "".join(
            f"{np.mean([r['stall_post_ttft'] for r in rows if r['scenario']==scen and r['policy']==p])*1000:17.1f} "
            for p in POL))
        ideal = rows[0]['decode_ideal']*1000
        print(f"   (ideal decode time for {len(need)} steps = {ideal:.0f} ms; "
              f"stall is mean over links)")
    print("\nwrote", a.out)

if __name__ == "__main__":
    main()
