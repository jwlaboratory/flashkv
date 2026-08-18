#!/bin/bash
set -u
P=.venv/bin/python
$P exp/e1_selection_trace.py --model Qwen/Qwen2.5-0.5B-Instruct --kind qa_needle \
   --ctx 16384 --steps 256 --block 64 --tail 4 --out results/e1l/q05_qa_16k 2>&1 | grep -v "it/s\]"
$P exp/e1_selection_trace.py --model Qwen/Qwen2.5-0.5B-Instruct --kind summarize \
   --ctx 16384 --steps 256 --block 64 --tail 4 --out results/e1l/q05_sum_16k 2>&1 | grep -v "it/s\]"
$P exp/e1_selection_trace.py --model Qwen/Qwen2.5-0.5B-Instruct --kind qa_needle \
   --ctx 32768 --steps 256 --block 64 --tail 4 --out results/e1l/q05_qa_32k 2>&1 | grep -v "it/s\]"
$P exp/e1_selection_trace.py --model Qwen/Qwen2.5-1.5B-Instruct --kind summarize \
   --ctx 16384 --steps 256 --block 64 --tail 4 --out results/e1l/q15_sum_16k 2>&1 | grep -v "it/s\]"
echo LONGDONE
