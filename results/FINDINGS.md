# Sparse-aware KV transfer for disaggregated prefill→decode — findings

**Verdict: the idea is half right, and the half that's wrong is the half it was pitched on.
A second, adversarial round (§7) found that the surviving half is itself dominated by a
stronger design — don't reorder the bulk transfer, mostly don't do it.**

The analogy to layer-wise pipelining does *not* carry over: in the fresh-prefill case
layer-wise already hides ~100% of the transfer, and sparse ordering saves **<0.1% of
TTFT**. But the same mechanism is a **3–12× TTFT win** in the case where there is no
prefill compute to hide behind — prefix-cache hits and KV reloads from a remote/host
tier. It also only works with a *shared-index* selector (DeepSeek DSA), not a per-head
one (NSA/MoBA/Quest), and only with a demand-pull path; push-only is worthless.

---

## 1. Sparsity survives — but only for shared-index selectors  (D2)

Oracle block selection measured from exact per-head attention on Qwen2.5-0.5B/1.5B,
11 traces, 4k–32k context, 4 prompt regimes, 16/64/128-token blocks.

| per-head budget | **per-head top-k** (NSA/MoBA/Quest)<br>KV that must be resident | **shared index** (DeepSeek DSA)<br>KV that must be resident |
|---|---|---|
| 1.6 % | 3.4 – 8.9 % | **1.6 %** |
| 6.25 % | 22 – 31 % | **6.3 %** |
| 12.5 % | 42 – 50 % | **12.5 %** |
| 25 % | 68 – 74 % | **25 %** |

Each head is sparse, but heads disagree, and the KV cache is shared — so what has to be
*resident* is the **union over heads**, which blows up ~4.5× at realistic budgets. At a
12.5% budget you already need half the cache; there is nothing meaningful left to defer.

A shared index (one score per key, all heads use the same selection — what DSA's
lightning indexer does, and what NSA does within a GQA group) makes the resident set
exactly the budget, by construction. **This is a hard architectural precondition.**

## 2. The deferred remainder really is deferrable  (D3)

Cumulative unique KV touched after 16 decode steps:

| selector | 16k ctx | 32k ctx |
|---|---|---|
| shared, 1.6 % budget | 4.9 % | 5.5 % |
| shared, 6.25 % budget | 20.4 % | 20.2 % |
| per-head, 6.25 % budget | 67 % | 64 % |

Step-to-step Jaccard for shared @1.6%: 0.52–0.94 (median ≈0.65). Working-set growth is
sublinear and barely moves with context length, so at 128k with DSA's 2048-token budget
~95% of the cache is genuinely cold for the first dozen-plus tokens. **This part of the
idea checks out.**

Larger blocks are markedly more stable (Jaccard 0.52 @16-tok → 0.64 @64-tok → 0.94
@128-tok blocks), and they also transfer more efficiently — the two pressures agree.

## 3. The prefill worker cannot know the selection, and guessing is not enough  (D1)

Selection is a function of the *decode* query, which lives on the decode worker. The only
query the prefill worker has is the last prefill token. Measured overlap between the last
prefill token's selection and the first decode token's: **0.61–0.96, median 0.76**.

That sounds usable. It is not, on its own:

| 128k ctx, 100GbE, prefix-cache hit | push-only | push + demand-pull (20 µs RTT) |
|---|---|---|
| layer-wise | 784 ms | 216 ms |
| sparse, predicted from last prefill token | **786 ms** | **67 ms** |
| sparse, oracle selection | 25 ms | 26 ms |

With 76% prediction accuracy and no way to ask for a miss, the decode worker waits for
the mispredicted blocks to arrive in ordinary stream order — i.e. it waits for everything,
and the whole scheme collapses to the baseline. **A demand-pull path is mandatory, not an
optimization.** Note also that pull alone gets layer-wise from 784→216 ms; prioritization
buys the remaining 216→67 ms. Both halves are needed.

A "zero-knowledge" prioritization (attention sinks + sliding window, which the prefill
worker *can* predict) covers 98% of the selected set at 4k context but only 59% at 16k and
42% at 32k — it decays with context and is worth ≤6% TTFT at 128k. Not a substitute.

## 4. Where the idea pays  (D5) — the decisive result

TTFT reduction of sparse-predicted vs. layer-wise, 128k ctx, DSA budget, push+pull:

| | 25GbE x-DC | 100GbE | RoCE 200Gb | IB NDR 400Gb | NVLink |
|---|---|---|---|---|---|
| **prefix-cache hit (no prefill)** | **66 %** | **69 %** | **64 %** | **58 %** | 29 % |
| prefill 20k tok/s | 0.1 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| prefill 5k tok/s | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| prefill 1k tok/s | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |

**The fresh-prefill row is the one the idea was pitched for, and it is flat zero.** At 128k
a layer of MLA KV is 151 MB — 12 ms on 100GbE — against ~430 ms of prefill compute per
layer. Layer-wise has 9–36× of headroom; the exposed transfer it leaves is 0.5–11 ms on
top of a 26 s prefill, and sparse ordering shaves a further 0.4–7.5 ms. Sparse ordering
does not reach 1% of TTFT until effective per-request bandwidth drops below **~0.1 GB/s**
(a 100GbE link shared ~125 ways).

Scaling in the cache-hit regime (100GbE): 8k → 19%, 32k → 64%, 128k → 68%, 512k → 69%.
Below ~8k on a fast link it is net **negative** (−11% on IB NDR): scattered small messages
cost more than they save. Benefit holds at 58–67% for budgets ≤3% of context, decaying to
41% at a 25% budget.

Cost side: deferring the remainder raises **post-TTFT stall** (trace-driven, 32k: 6.8 → 20.7 ms
over 16 steps). It's a TTFT-for-TPOT trade, net positive here but not free.

## 5. Transfer granularity is a real but manageable tax  (D4)

Measured on this host, plus an analytic RDMA model with published constants:

- **Device-side gather is nearly free** — MPS `index_select` runs at 0.83–0.90× of a
  contiguous copy across 4 KB–1 MB blocks. Building the scatter list is not the problem.
- **Host-side gather** collapses at small granularity: 0.13× at 4 KB, 0.71× at 64 KB,
  ~1.0× at ≥1 MB.
- **Wire overhead**: a 64 KB message reaches only 47–64% of peak on IB/RoCE/100GbE
  (25% on NVLink); ≥1 MB reaches 84–97%.

A 64-token MLA block is 72 KB per layer — right in the bad zone at ~50–60% of peak. The fix
is to coalesce ≥16 blocks per message (≥1 MB), which the simulator already assumes. This
also argues for larger selection blocks, which §2 shows are more stable anyway.

---

## 6. Yes — the selection is predictable, and over-fetching is nearly free  (follow-up to D1)

§3 showed a push-only schedule collapses at 76% prediction accuracy. That framing was too
pessimistic in one respect: you do not have to predict *exactly* k blocks. Ranking blocks
and sending the top **m·k** costs almost nothing at a DSA budget — 4× of 1.56% is still
only 6% of the cache.

Recall of the blocks the **first decode token** actually reads (4 traces, 16–32k ctx,
DSA-proportional budget):

| predictor | who has it | m=1 | m=2 | m=4 | m=8 |
|---|---|---|---|---|---|
| **last prefill token** | prefill worker, free | **0.733** | 0.859 | **0.924** | **0.977** |
| mean of last 8 prefill tokens | prefill worker, free | 0.689 | 0.812 | 0.908 | 0.964 |
| previous decode step | decode worker | 0.733 | 0.859 | 0.924 | 0.977 |
| previous layer, same step | decode worker, JIT | 0.617 | 0.715 | 0.824 | 0.891 |
| layer 0's selection, reused | decode worker | 0.447 | 0.579 | 0.629 | 0.658 |
| sinks + sliding window | query-independent | 0.438 | 0.604 | 0.607 | 0.628 |
| random | — | 0.036 | 0.004 | 0.068 | 0.126 |

Findings:

- **The last prefill token is the best predictor available, and it is free.** 73% exact,
  92% at 4× over-fetch, 98% at 8×. Nothing beats it — averaging over the last 8 prefill
  positions is slightly *worse* for the first token (the selection tracks the most recent
  query, it is not a stable "popularity" property).
- **In steady state the previous decode step is better still** (0.926 at m=4 averaged over
  all steps vs 0.791 for last-prefill), so the predictor should switch to prev-step once
  decoding is underway.
- **Cross-layer reuse does not work.** Layers agree only moderately (mean pairwise Jaccard
  0.47), and reusing layer 0's selection for every later layer gets 0.63 at m=4 — worse
  than the free prefill-side predictor. The just-in-time variant (use layer ℓ−1's true
  selection to prefetch layer ℓ, which is timing-feasible) reaches 0.82, still worse. Each
  layer needs its own prediction.
- **Sinks + sliding window saturate at ~0.60** and do not improve with over-fetch — it is a
  fixed set, not a ranking. Confirms §3: not a substitute.

### What the over-fetch is worth (128k, DeepSeek geometry, prefix-cache hit, 6 seeds)

| over-fetch | priority set | recall | TTFT 25GbE | TTFT 100GbE | TTFT IB NDR |
|---|---|---|---|---|---|
| *layer-wise + pull* | — | — | 693 ms | 216 ms | 98 ms |
| 1× | 1.56 % | 0.750 | 260 ms | **75 ms** | 45 ms |
| 2× | 3.12 % | 0.868 | 196 ms | 77 ms | 35 ms |
| 4× | 6.25 % | 0.938 | 159 ms | 86 ms | **31 ms** |
| 6× | 9.38 % | 0.960 | **143 ms** | 82 ms | 67 ms |
| 8× | 12.5 % | 0.979 | 144 ms | 82 ms | 67 ms |

**The slower the link, the more you should over-fetch** — a miss costs a round trip plus a
transfer, and on a slow link that dominates the cost of shipping extra blocks (25GbE:
260 → 143 ms going from 1× to 6×). On fast links the priority wave itself starts competing
with the demand pulls for the channel and the curve turns back up (IB NDR: best at 4×,
then a cliff). Run-to-run spread is ≤2 ms, so this is a real scheduling effect, not noise —
though where exactly the cliff falls is model-dependent and should not be read as precise.

Over-fetch also consistently reduces **post-TTFT stall** (100GbE, 12 steps: 260 ms at 1× →
223 ms at 8×), so it buys back part of the TPOT cost noted in §4.

**Practical rule: rank by the last prefill token, send 2–4× the budget, switch to
previous-decode-step ranking once decoding starts, keep a demand-pull path for the ~6–8%
you miss.**

---

## 7. Adversarial round: the premise "we still send everything" is the part that fails

The idea as posed keeps the bulk transfer and only reorders it. Two measurements say that
is the wrong design — the cold blocks should mostly never be sent at all.

### The working set saturates (256-step traces, the measurement §2 was too short to make)

| trace | WS@8 | WS@32 | WS@128 | WS@256 | fresh blocks/step, last 32 |
|---|---|---|---|---|---|
| Qwen0.5B qa 16k | 3.7 % | 8.1 % | 14.2 % | **16.3 %** | 0.1 % |
| Qwen0.5B qa 32k | 3.9 % | 8.2 % | 13.5 % | **16.9 %** | 0.6 % |
| Qwen0.5B summarize 16k | 2.9 % | 7.2 % | 12.0 % | **12.8 %** | 0.8 % |
| Qwen1.5B summarize 16k | 2.6 % | 5.9 % | 12.3 % | **14.7 %** | 1.2 % |

Generating 256 tokens touches ~15% of the KV, and the marginal rate has fallen to ~1% —
the selection keeps returning to blocks it has already used. Extrapolating the tail slope,
1024 generated tokens still only reaches ~24%. **A bulk transfer ships 4–7× more data than
the request will ever read**, and there is no generation length at which it breaks even.

*(This also invalidated the naive synthetic trace generator, which random-walks to 54%
coverage by step 256. `synth_ws` in `e2_pipeline_sim.py` is calibrated to the measured
fresh-block curve and reproduces 15.1% vs the measured 15.2%. Earlier long-horizon
synthetic numbers were inflated.)*

### So paging beats priority-ordering on every axis (128k, 256 tokens, 20 µs RTT)

| 100GbE | TTFT | TPOT | GB moved |
|---|---|---|---|
| layer-wise bulk | 216 ms | 25.8 ms | 9.21 (100 %) |
| sparse priority + bulk | 67 ms | 26.0 ms | 9.21 (100 %) |
| **paged + prefetch** | **33 ms** | **25.0 ms** (ideal) | **1.42 (15 %)** |
| paged, cold index cache | 113 ms | 25.0 ms | 2.44 (27 %) |

Priority-ordering is *worse* than paging on TPOT precisely because the background stream
competes with demand fetches for the channel. And capacity moves with it: a bulk transfer
forces the whole 9.21 GB resident on the decode worker, a pager only the working set plus
index — 17 vs 4 concurrent 128k requests on an 80 GB H100, and 104 vs 27 on 100GbE.

### Two things that bound paging

**1. Round-trip latency.** A pager stalls on a request whenever it wants a block it does
not hold; misses batch within a layer, so the worst case is 1 RTT per layer per token —
61× for DeepSeek-V3.2. TPOT (ideal = 25 ms):

| RTT | 61×RTT | layer-wise bulk | paged + prefetch |
|---|---|---|---|
| 20 µs | 1.2 ms | 28.8 ms | **25.0 ms** |
| 50 µs | 3.0 ms | 29.7 ms | **25.0 ms** |
| 100 µs | 6.1 ms | **31.3 ms** | 36.1 ms |
| 200 µs | 12.2 ms | **34.5 ms** | 69.3 ms |
| 1 ms | 61 ms | **60.2 ms** | 334.8 ms |

**Paging wins below ~100 µs RTT and loses above it.** In-datacenter RDMA (2–10 µs, or
20–50 µs through a software stack) is comfortably inside. Cross-region is not — there,
bulk transfer is correct and the original priority-ordering idea is the right one.

**2. The index cache is on the critical path.** A DSA pager cannot select a block until it
holds the lightning-indexer keys for *every* token — 128 B/token/layer in fp8, 11% of the
KV, 1.02 GB at 128k. If that has to be cold-fetched, paging's TTFT goes 33 → 113 ms on
100GbE and 104 → 444 ms on 25GbE, **losing to sparse-priority+bulk** (67 ms / 232 ms). The
index cache must be pre-staged or kept resident; it is 11% of the KV, so this is cheap to
do deliberately and fatal to ignore.

### Revised design

Keep the **index cache resident** (11%, mandatory, predictable — no prediction needed).
**Page the MLA latents** on demand with last-prefill-token prefetch. Total ~26% of the KV
moved for a 256-token generation, TTFT ~33 ms, TPOT at the ideal, 4× the concurrency.
Fall back to sparse-priority bulk transfer when RTT to the KV store exceeds ~100 µs.

**What survives of the original idea:** the sparsity insight, the prediction mechanism, and
the priority ordering — all of which the pager reuses as its prefetcher. **What does not:**
"of course we'll still send everything over." That is the assumption to drop.

---

## What to build, if you pursue it

1. Target the **prefix-cache-hit / KV-tier-load** path, not fresh prefill. Same mechanism,
   the regime where it actually pays.
1b. **Do not bulk-transfer the cold blocks at all** (§7) — page them. Keep the indexer-key
   cache resident. Revert to priority-ordered bulk only if RTT to the store is >~100 µs.
2. Requires a **shared-index (DSA-style) selector**. Per-head selection kills it.
3. **Push + demand-pull**, not push alone. Rank blocks by the last prefill token's own
   attention and push 2–4× the budget (§6): that reaches 86–92% recall for a priority set
   of only 3–6% of the cache. Switch to previous-decode-step ranking once decoding starts
   (0.93 recall at 4×). The remaining ~8% must have a pull path.
4. Coalesce to ≥1 MB messages; prefer ≥64-token selection blocks.
5. Watch TPOT — the deferred stream must be drained aggressively or later tokens stall.

## What these experiments do NOT establish

- Oracle selection is taken from **true attention mass on dense-trained** Qwen2.5 models,
  not from a trained sparse selector, at **16–32k**, at **0.5B/1.5B**. A DSA-trained 671B
  model at 128k should be *more* concentrated and more stable, so this likely understates
  sparsity — but it is an extrapolation, not a measurement.
- No real two-node RDMA test. The link is an analytic model (per-message overhead +
  message-rate cap) calibrated with published NIC constants; the gather numbers are
  measured, but on Apple silicon, not on an H100/H800 + CX-7.
- Prefill throughput and decode step time are parameters, swept rather than measured.

- The 256-step working-set curves are the load-bearing measurement of §7 and they come from
  16–32k contexts. Saturation at 128k is an extrapolation of the tail slope.
- The RTT crossover (~100 µs) comes from a model in which misses batch per layer. A real
  implementation that fails to batch them would break far earlier; one that prefetches
  further ahead would push it later.

**The experiment that would settle it**: DeepSeek-V3.2-Exp on two nodes, measure TTFT/TPOT
for a prefix-cache hit with (a) layer-wise push, (b) layer-wise push + demand-pull,
(c) DSA-index priority push + demand-pull, (d) pure paging with resident index cache.
Prediction from this work: (a) ≫ (b) > (c) > (d) on TTFT, with (d) moving ~5× less data and
holding TPOT at the compute floor, provided RTT stays under ~100 µs.

---

### Reproducing

```bash
./exp/run_e1_sweep.sh                       # attention traces (~15 min)
.venv/bin/python exp/e1a_analyze.py         # §1, §2, §3 tables
.venv/bin/python exp/e3_granularity.py      # §5
.venv/bin/python exp/e2_pipeline_sim.py     # §4, DeepSeek-scale synthetic
.venv/bin/python exp/e2_pipeline_sim.py --mode trace --trace results/e1/q05_qa_32k_b64
.venv/bin/python exp/e5_regime.py           # §4 regime map + crossover
.venv/bin/python exp/e6_plots.py            # results/findings.png
.venv/bin/python exp/run_e1_tail.sh         # traces w/ 8 prefill tail positions
.venv/bin/python exp/e7_predictors.py --budget 0.0156   # §6 predictor table
.venv/bin/python exp/e8_overfetch.py        # §6 over-fetch sweep
.venv/bin/python exp/e9_pred_plots.py       # results/predictors.png
./exp/run_e1_long.sh                        # 256-step traces (~25 min)
.venv/bin/python exp/e10_longgen.py         # §7 working-set saturation
.venv/bin/python exp/e11_paging.py          # §7 paging, RTT attack, index cache
.venv/bin/python exp/e12_capacity.py        # §7 bandwidth + HBM capacity
.venv/bin/python exp/e13_plots2.py          # results/paging.png
```
