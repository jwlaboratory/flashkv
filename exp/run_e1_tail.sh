#!/bin/bash
set -u
P=.venv/bin/python
for K in qa_needle summarize code; do
  $P exp/e1_selection_trace.py --model Qwen/Qwen2.5-0.5B-Instruct --kind $K \
     --ctx 16384 --steps 24 --block 64 --tail 8 --out results/e1t/q05_${K}_16k 2>&1 | grep -v "it/s\]"
done
$P exp/e1_selection_trace.py --model Qwen/Qwen2.5-0.5B-Instruct --kind qa_needle \
   --ctx 32768 --steps 24 --block 64 --tail 8 --out results/e1t/q05_qa_32k 2>&1 | grep -v "it/s\]"
$P exp/e1_selection_trace.py --model Qwen/Qwen2.5-1.5B-Instruct --kind qa_needle \
   --ctx 16384 --steps 24 --block 64 --tail 8 --out results/e1t/q15_qa_16k 2>&1 | grep -v "it/s\]"
echo TAILDONE
