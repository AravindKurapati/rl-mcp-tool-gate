#!/usr/bin/env bash
set -e
export PYTHONIOENCODING=utf-8
mkdir -p results
if [ ! -f results/qwen_baseline_preds.jsonl ]; then
  modal run src/eval/qwen_baseline.py::predict
fi
python -m src.eval.pareto
