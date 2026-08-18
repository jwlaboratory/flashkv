# flashkv

Viability probe: **can block-sparse attention (DeepSeek DSA / NSA / MoBA) be used to
prioritise KV-cache transfer from a prefill worker to a decode worker** — sending the
blocks the first decode token needs first, then lazily streaming the rest, the way
layer-wise pipelining sends layer 0 first?

**Short answer: not for fresh prefill (layer-wise already hides ~all of it — <0.1% of
TTFT left to win), but yes for prefix-cache hits / KV reloads, where it is a 3–12× TTFT
win. It requires a shared-index selector and a demand-pull path.**

- [`PLAN.md`](PLAN.md) — the falsifiable structure: five ways the idea could die
- [**`results/FINDINGS.md`**](results/FINDINGS.md) — results and verdict
- `results/findings.png` — four decisive plots

| experiment | question |
|---|---|
| `exp/e1_selection_trace.py` | oracle block selection traces from real LLMs |
| `exp/e1a_analyze.py` | is the critical set small, stable, predictable? |
| `exp/e2_pipeline_sim.py` | event-driven P→D transfer sim, 6 policies |
| `exp/e3_granularity.py` | what does scattered block transfer cost? |
| `exp/e5_regime.py` | *where* does the idea pay? |
| `exp/e6_plots.py` | figures |
