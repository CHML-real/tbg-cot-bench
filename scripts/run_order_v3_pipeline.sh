#!/usr/bin/env bash
set -e

echo "[1/5] Run EXAONE order-v3 extraction..."
OLLAMA_MODEL="${OLLAMA_MODEL:-exaone-local}" python scripts/run_ollama_order_v3.py

echo "[2/5] Parse order-v3 outputs..."
python scripts/parse_ollama_order_v3.py

echo "[3/5] Compute order-v3 trajectories..."
python scripts/run_order_v3_trajectory.py

echo "[4/5] Evaluate order-v3..."
python scripts/evaluate_order_v3.py

echo "[5/5] Visualize order-v3 comparison..."
python scripts/visualize_order_v3_comparison.py

echo "Done."
echo "Check:"
echo "  results/order_v3_eval_summary.csv"
echo "  figures/order_v3_accuracy_comparison.png"
echo "  figures/order_v3_parse_success_comparison.png"
