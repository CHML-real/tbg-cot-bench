#!/usr/bin/env bash
set -e

echo "[1/5] Resume missing/failed EXAONE step-wise calls..."
OLLAMA_MODEL="${OLLAMA_MODEL:-exaone-local}" python scripts/run_ollama_stepwise_evidence_v21.py --mode failed

echo "[2/5] Parse step-wise outputs with robust parser..."
python scripts/parse_ollama_stepwise_evidence_v21.py

echo "[3/5] Recompute step-wise trajectories..."
python scripts/run_stepwise_trajectory.py

echo "[4/5] Evaluate step-wise EXAONE..."
python scripts/evaluate_stepwise_ollama.py

echo "[5/5] Visualize step-wise comparison..."
python scripts/visualize_stepwise_comparison.py

echo "Done."
echo "Check:"
echo "  results/stepwise_ollama_eval_summary.csv"
echo "  figures/stepwise_vs_baseline_accuracy.png"
echo "  figures/stepwise_parse_success.png"
