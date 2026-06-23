#!/usr/bin/env python
from __future__ import annotations
import argparse, csv
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tbg_cot_bench.core import load_scenarios, simulate_gold, summarize_trajectory, DEFAULT_LEARNING_RATE


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gold-label TBG-CoT trajectories.")
    parser.add_argument("--scenarios", default="scenarios")
    parser.add_argument("--out", default="results")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    args = parser.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios(args.scenarios)
    all_rows = []
    summaries = []
    for scenario in scenarios:
        rows = simulate_gold(scenario, learning_rate=args.learning_rate)
        all_rows.extend(rows)
        summaries.append(summarize_trajectory(scenario, rows))

    with (out / "trajectories_gold.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader(); writer.writerows(all_rows)
    with (out / "scenario_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader(); writer.writerows(summaries)
    print(f"Wrote {out / 'trajectories_gold.csv'} and {out / 'scenario_summary.csv'}")

if __name__ == "__main__":
    main()
