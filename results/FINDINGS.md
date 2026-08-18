# Sparse-aware KV transfer for disaggregated prefill→decode — findings

**Verdict: the idea is half right, the half that's wrong is the half it was pitched on,
the surviving half is dominated by a stronger design (§7 — don't reorder the bulk transfer,
mostly don't do it), and that stronger design is already published (§9 — HiSparse, SAC).
§8 corrects a 25x error in the prefill-throughput assumption; the conclusion survives it.
§10 finds the real predictive edge: one-step lookahead prefetch breaks the 61-layer serial
round-trip chain and moves the latency wall from ~650us to >60ms, making paging viable
cross-region. Perfect prediction adds only ~25% on top of the measured 0.93 predictor.**

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

## 8. CORRECTION: the prefill-throughput number was wrong by ~25x

§4 assumed a prefill instance sustains 5,000 tok/s, giving 26 s to prefill 128k. **That is
roughly the per-GPU figure, not the per-instance figure.** DeepSeek's published V3 inference
system uses a 4-node / 32-GPU prefill unit.

Regrounding: 73.7k tok/s per H800 node at a 56.3% prefix-cache-hit rate implies ~32.2k
*computed* tok/s per node, ~4.0k tok/s per GPU. At 74 GFLOP/token (2 x 37B active) that is
~298 TFLOPS/GPU, ~30% MFU on fp8 H800 — self-consistent. So a 32-GPU prefill instance does
128k tokens in **~1.0 s, not 26 s**, and there is 25x less compute to hide the transfer
behind than §4 assumed.

Whether the transfer still hides (128k = 9.21 GB of MLA KV):

| prefill GPUs | tok/s | 128k prefill | KV egress | full xfer @25GbE / 100GbE / IB400 |
|---|---|---|---|---|
| 8 | 32k | 4.10 s | 2.2 GB/s | 2.97 / 0.74 / 0.18 s — hidden everywhere |
| 32 (DeepSeek's unit) | 128k | **1.02 s** | 9.0 GB/s | 2.97 / 0.74 / 0.18 s — 25GbE transfer-bound |
| 64 | 256k | 0.51 s | 18.0 GB/s | 25GbE **and** 100GbE transfer-bound |

So the headroom is real but far from the 9–36x §4 claimed. **Does the conclusion survive?**
Fresh prefill, 32-GPU instance, absolute TTFT (prefill compute floor = 1024 ms):

| link | layer-wise **push** | layer-wise **+pull** | sparse **+pull** | saved vs push | vs pull |
|---|---|---|---|---|---|
| 25GbE | 3145 ms | 1036 ms | 1028 ms | 67.3 % | **0.7 %** |
| 100GbE | 1037 ms | 1028 ms | 1026 ms | 1.1 % | **0.2 %** |
| IB NDR 400Gb | 1028 ms | 1026 ms | 1025 ms | 0.3 % | **0.0 %** |
| NVLink | 1025 ms | 1025 ms | 1025 ms | 0.0 % | 0.0 % |

**The §4 conclusion survives the correction, but the reason shifts.** At 25GbE the transfer
genuinely is on the critical path now (3145 ms vs a 1024 ms floor) — but the fix that
recovers it is **demand-pull** (3145 → 1036 ms, 67%), not sparse ordering, which adds a
further 0.7%. Above 100GbE everything is within ~1% of the prefill floor. Sparse priority
ordering is still worth ≤1% of TTFT for fresh prefill at every realistic operating point;
only an unrealistic corner (128-GPU prefill instance on 25GbE) reaches 63%.

## 9. Prior work: most of this exists, and the numbers agree

The paging design §7 converged on is published, and recently.

- **HiSparse** ([arXiv 2608.07009](https://arxiv.org/html/2608.07009)) is the closest. Two-tier
  KV hierarchy: full history in pinned host DRAM, a fixed-size per-request GPU cache with
  LRU, a fused kernel doing hit-detect / replace / batched H2D fetch inside the decode CUDA
  graph, plus layer-wise prefetch for models that share selections across layers. Evaluated
  on DSA (GLM-5.1/5.2, k=2048), NSA (DeepSeek-V4-Flash) and Quest, on H200/B200/GH200. Up
  to 4.7x throughput at 200k context.
  **Their headline locality number is ours.** They report an 87% hit rate with a GPU cache
  of 2k slots — twice the selection size. Our E7 over-fetch curve gives 0.868 recall at
  m=2. Independently measured, different models, same number. Their finding that LRU
  tracks Bélády is our "previous decode step is the best steady-state predictor" (0.93 at
  m=4).
- **SAC** ([arXiv 2606.19746](https://arxiv.org/html/2606.19746v1)) puts exactly §7's design in the
  disaggregated setting: full KV in a CXL pool, reactive sub-block fetching driven by
  layer-wise top-k, on DeepSeek-V3.2 across 8x H20 with a 2TB CXL pool. 9.7x lower TTFT and
  2.1x throughput vs an RDMA baseline, within 9% of local DRAM.
  **Their RTT result is our §7 bound.** They measure CXL at 1.04–1.64x local-DRAM latency
  versus RDMA at 4.0–19.7x, and conclude load/store semantics are what make sparse paging
  viable. That is the same effect as our ~100 µs RTT crossover, measured on real hardware.
- **InfiniGen** (OSDI'24) is the ancestor of the prefill-side predictor: speculative
  prefetch of important KV entries from CPU using SVD projections. **ShadowKV** keeps
  low-rank keys on GPU and fetches values on demand. **ArkVale** evicts cold pages and
  recalls them by page summary.
- On the P→D transfer path specifically: **SpectrumKV** (per-token mixed-precision KV
  transfer, importance-scored at the prefill worker before the wire), **PDTrim** (~5x less
  cross-node transfer via token/layer-selective pruning), **Semantic Cache Distillation**
  (low-rank reconstruction, 2.65x TTFT).

**What is left.** HiSparse's LRU and SAC's reactive fetch both need history, so neither
helps the *first* decode token after a cache hit — SAC is explicitly "purely demand-driven,"
which means the first token pays a serial fetch per layer. Our §6 result — the last prefill
token's own attention predicts the first decode token's selection at 0.73 exact / 0.92 at
4x over-fetch, and 4x is only 6% of the cache — is a cold-start prefetch hint that neither
system uses. That is a narrow contribution on top of an occupied field, not a new direction.

---

## 10. CORRECTED: one-step lookahead prefetch is the edge, and it is a large one

*An earlier version of this section reported that lookahead prefetch buys nothing. That was
a simulator bug: `pop_ready()` removed a cold unit from the heap and the skip path then
discarded it permanently, so every cold unit drained out before any prefetch flag was set
and **the prefetch never transferred a single block**. Fixed (explicit block→unit lookup,
cold units enter the queue only via demand or prefetch). The corrected result is the
opposite of what that section claimed.*

### Why speculation matters: it breaks the serial chain

Layer 0 needs no prediction at all — its query is just the token embedding, so with a
resident index cache the decode worker computes layer 0's exact top-k instantly. The
problem is that layer l+1's query depends on layer l's output, so a reactive pager walks a
**61-long serial chain**, paying a round trip at each link. Bootstrapping layer 0 alone does
not help. Predicting *all* layers one step ahead removes the chain entirely: every layer's
fetch is issued a whole token-time (~25 ms) before it is needed.

### It works, and one step is enough

TPOT, 128k, 100GbE, prefix-cache hit (ideal 25.0 ms):

| RTT | bulk | reactive (h=0) | **h=1** | h=2 | h=4 | h=16 |
|---|---|---|---|---|---|---|
| 20 µs | 30.8 | 25.0 | **25.0** | 25.1 | 25.3 | 25.8 |
| 500 µs | 39.0 | 32.6 | **25.7** | 26.6 | 28.5 | 35.8 |
| 1 ms | 47.5 | 63.1 | **26.5** | 28.1 | 31.9 | 48.1 |
| 5 ms | 115.8 | 307.1 | **37.9** | 47.5 | 67.6 | 155.5 |
| 20 ms | 371.8 | 1222.1 | **99.7** | 135.6 | 217.2 | 567.5 |

**The latency wall moves from ~650 µs to beyond 60 ms** — at 20 ms RTT paging is still 3.7×
better than bulk transfer. That is cross-region territory, not just cross-rack.

**h=1 is the sweet spot; deeper is worse.** h=16 is 5.7× worse than h=1 at 20 ms RTT,
because deeper horizons have lower recall, so more of the speculative fetch is junk that
competes for bandwidth — the same "speculative traffic crowds out needed traffic" effect
seen in §6 and §7. The lesson survives; it just bounds the depth rather than the idea.

Cost: prefetch converts round trips into bytes. Over 24 steps at 2× over-fetch, h=1 moves
3.58 GB vs 0.73 GB reactive — still well under the 9.21 GB a bulk transfer sends.

### How accurate does the predictor have to be? Less than you would think

TPOT vs per-block recall of the prefetch set, with h=1 (bulk shown for comparison):

| per-block recall | layers still stalling | RTT 1 ms | RTT 5 ms |
|---|---|---|---|
| 0.900 | 58.9 / 61 | 28.1 | 52.0 |
| 0.930 *(measured, 2× over-fetch)* | 55.0 / 61 | 27.0 | 41.6 |
| 0.959 *(measured, 4×)* | 45.0 / 61 | 26.8 | 36.0 |
| 0.990 | 16.8 / 61 | 26.3 | 31.3 |
| **1.000 (perfect)** | 0 / 61 | **26.3** | **31.3** |
| bulk transfer | — | 55.4 | 147.4 |

Perfect top-k prediction is worth ~25% over the measured 0.93 predictor at 5 ms RTT, and
almost nothing at 1 ms. **The unlock is having *a* prediction plus one step of lookahead,
not having a perfect one** — 0.90 recall already beats bulk by 2.8× at 5 ms RTT. Chasing
accuracy past ~0.96 is not where the value is.

Note the counter-intuitive column: even at recall 1.0 the naive formula says 0 layers stall,
yet TPOT is 26.3 ms rather than 25.0 — the residue is bandwidth, not latency. Once round
trips are pipelined away, the pager is limited by moving the bytes, which is the regime you
want to be in.

### Caveat

The simulator issues all 61 layers' prefetches at the start of step t. A real
implementation learns layer l's selection only as step t reaches layer l, so early layers
get somewhat less slack than modelled, and the effective predictor is closer to horizon 2
than horizon 1. The h=2 row (28.1 ms at 1 ms RTT, 47.5 at 5 ms) bounds that, and the
conclusion holds comfortably.

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
.venv/bin/python exp/e14_realistic.py       # §8 corrected hardware numbers
.venv/bin/python exp/e15_horizon.py         # §10 prediction vs lookahead horizon
.venv/bin/python exp/e16_lookahead.py       # §10 does lookahead move the latency wall?
```
