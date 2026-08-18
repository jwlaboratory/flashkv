#!/bin/bash
set -u
P=.venv/bin/python
M05=Qwen/Qwen2.5-0.5B-Instruct
M15=Qwen/Qwen2.5-1.5B-Instruct
run(){ echo "=== $* ==="; $P exp/e1_selection_trace.py "$@" 2>&1 | grep -v "it/s\]" ; }

# main matrix: 4 prompt regimes at 16k on 0.5B
for K in qa_needle summarize code continue; do
  run --model $M05 --kind $K --ctx 16384 --steps 32 --block 64 --budget 1024 --out results/e1/q05_${K}_16k_b64
done
# block-size sensitivity
for B in 16 128; do
  run --model $M05 --kind qa_needle --ctx 16384 --steps 32 --block $B --budget 1024 --out results/e1/q05_qa_16k_b${B}
done
# context-length sensitivity
run --model $M05 --kind qa_needle --ctx 4096  --steps 32 --block 64 --budget 1024 --out results/e1/q05_qa_4k_b64
run --model $M05 --kind qa_needle --ctx 32768 --steps 32 --block 64 --budget 1024 --out results/e1/q05_qa_32k_b64
# second model
for K in qa_needle summarize; do
  run --model $M15 --kind $K --ctx 16384 --steps 32 --block 64 --budget 1024 --out results/e1/q15_${K}_16k_b64
done
echo ALLDONE
