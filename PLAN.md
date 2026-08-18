# Sparse-aware KV transfer for disaggregated prefill/decode — viability probe

## The idea under test

In disaggregated serving, prefill (P) hands the KV cache to decode (D).
- **Blocking**: finish all prefill, ship all KV, start decode.
- **Layer-wise** (state of the art): ship layer L's KV as soon as P finishes layer L, overlapping transfer with the rest of prefill compute.
- **Proposed**: with block-sparse attention (DeepSeek DSA / NSA / MoBA / Quest), a decode step at layer L reads only `k` of `N` KV blocks. So ship *those blocks first*, stream the remainder lazily in the background. Everything still gets sent; only the order changes.

## What would make this true (and how each part can die)

| # | Claim it needs | How it dies | Experiment |
|---|---|---|---|
| **D1** | P can know *which* blocks D will need | Selection is a function of the **decode query**, which lives on D. P cannot know it without a round trip. | E1 (predictability from last prefill token), E4 (RTT cost of the pull-based variant) |
| **D2** | The per-layer critical set is small | Per-head selections diverge; the **union** over heads is most of the cache even though each head is sparse. Then "priority" ≈ "everything". | E1 |
| **D3** | The deferred remainder is genuinely not needed soon | Later decode steps touch fresh blocks; the working set grows fast; you stall on misses instead of saving time. | E1 (working-set growth, step-to-step stability) + E2 (stall simulation) |
| **D4** | Scattered block transfer keeps line rate | Priority order = scatter-gather of many small messages. If per-message overhead dominates, sending 2% scattered costs as much as 40% contiguous. | E3 |
| **D5** | There is headroom left to win | Layer-wise already hides transfer under prefill compute. If transfer is not the bottleneck, reordering it saves **nothing**. The idea may be correct but useless. | E2 (roofline / regime map) |

**Falsification bar.** Any one of: critical-set union > ~50% of cache (D2); step-to-step selection overlap so low that working set → 100% within a few steps (D3); or E2 showing layer-wise already at ~0 exposed transfer across realistic bandwidths (D5) — kills or heavily narrows the idea.

## Experiments

- **E1 `exp/e1_selection_trace.py`** — Oracle block-selection traces from real LLMs. Exact per-head attention at each decode step over a long context; block scores; top-k / mass-covering selections. Measures: per-layer critical-set size, head-union blowup, step-to-step overlap, working-set growth, and last-prefill-token → first-decode-token predictability. *Oracle selection is the upper bound: if the oracle is not sparse/stable, no trained selector will be.*
- **E2 `exp/e2_pipeline_sim.py`** — Discrete-event simulator of P→D transfer under three policies (blocking / layer-wise / sparse-priority), driven by E1 traces. Sweeps bandwidth, context length, model size. Reports TTFT and decode stall time. Produces the regime map for D5.
- **E3 `exp/e3_granularity.py`** — Achieved bandwidth vs. message size and scatter degree, to price the scatter-gather penalty (D4).
- **E4 `exp/e4_roundtrip.py`** — Latency model for the pull-based variant: D computes the index, requests blocks, waits RTT. Does the round trip eat the win? (D1)

## Status
See `results/FINDINGS.md`.
