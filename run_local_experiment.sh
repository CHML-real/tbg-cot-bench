#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install matplotlib

python scripts/run_gold_trajectory.py --scenarios scenarios --out results --learning-rate 0.4
python scripts/run_auto_converter.py --scenarios scenarios --out results --learning-rate 0.4
python scripts/evaluate_converter.py --scenarios scenarios --out results
python scripts/visualize_results.py --results results --figures figures

echo ""
echo "Done. Check:"
echo "  results/"
echo "  figures/"
