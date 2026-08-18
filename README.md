# flashkv

Viability probe: **can block-sparse attention (DeepSeek DSA / NSA / MoBA) be used to
prioritise KV-cache transfer from a prefill worker to a decode worker** — sending the
blocks the first decode token needs first, then lazily streaming the rest, the way
layer-wise pipelining sends layer 0 first?

**Short answer: no for fresh prefill — layer-wise already hides ~all of it, leaving <0.1%
of TTFT to win. Yes for prefix-cache hits / KV reloads (3–12× TTFT). But an adversarial
round then found the surviving version is itself dominated: 256-token generations touch
only ~15% of the KV, so the right design is to *not send the cold blocks at all* and page
them on demand — 33 ms TTFT, ideal TPOT, 15% of the bytes, 4× the concurrency. That holds
below ~100 µs RTT and requires the DSA indexer-key cache to be resident.

That design is already published — see §9: HiSparse (two-tier sparse KV hierarchy) and SAC
(the same over a CXL pool, on DeepSeek-V3.2) report the same locality and latency numbers
we measured independently.**

- [`PLAN.md`](PLAN.md) — the falsifiable structure: five ways the idea could die
- [**`results/FINDINGS.md`**](results/FINDINGS.md) — results and verdict
- `results/findings.png`, `results/predictors.png`, `results/paging.png` — plots

| experiment | question |
|---|---|
| `exp/e1_selection_trace.py` | oracle block selection traces from real LLMs |
| `exp/e1a_analyze.py` | is the critical set small, stable, predictable? |
| `exp/e2_pipeline_sim.py` | event-driven P→D transfer sim, 6 policies |
| `exp/e3_granularity.py` | what does scattered block transfer cost? |
| `exp/e5_regime.py` | *where* does the idea pay? |
| `exp/e7_predictors.py` | can we predict which blocks are needed? |
| `exp/e8_overfetch.py` | how much should we speculatively over-fetch? |
| `exp/e10_longgen.py` | does the working set saturate over a long generation? |
| `exp/e11_paging.py` | bulk vs paging; the RTT attack; the index-cache floor |
| `exp/e12_capacity.py` | bandwidth + HBM capacity at fleet scale |
| `exp/e14_realistic.py` | corrected prefill throughput (§8) |
| `exp/e6_plots.py` | figures |
