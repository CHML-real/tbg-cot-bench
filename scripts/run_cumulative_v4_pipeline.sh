#!/usr/bin/env bash
set -e

echo "[1/4] Run cumulative-v4 EXAONE extraction..."
OLLAMA_MODEL="${OLLAMA_MODEL:-exaone-local}" python scripts/run_ollama_cumulative_v4.py

echo "[2/4] Parse cumulative-v4 outputs..."
python scripts/parse_ollama_cumulative_v4.py

echo "[3/4] Evaluate cumulative-v4 trajectory..."
python scripts/evaluate_cumulative_v4.py

echo "[4/4] Visualize cumulative-v4 comparison..."
python scripts/visualize_cumulative_v4.py

echo "Done."
echo "Check:"
echo "  results/cumulative_v4_eval_summary.csv"
echo "  figures/cumulative_v4_accuracy_comparison.png"
