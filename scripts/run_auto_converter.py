#!/usr/bin/env python
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tbg_cot_bench.core import load_scenarios, simulate_evidence_steps, summarize_trajectory, DEFAULT_LEARNING_RATE
from tbg_cot_bench.converter_baseline import EvidenceConverter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline auto-converter on benchmark scenarios.")
    parser.add_argument("--scenarios", default="scenarios")
    parser.add_argument("--out", default="results")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    args = parser.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    converter = EvidenceConverter()
    scenarios = load_scenarios(args.scenarios)
    evidence_rows = []
    trajectory_rows = []
    for scenario in scenarios:
        event_a = scenario["events"]["event_a"]["label"]
        event_b = scenario["events"]["event_b"]["label"]
        parsed = converter.convert_steps([s["text"] for s in scenario["steps"]], scenario_id=scenario["id"], event_a_label=event_a, event_b_label=event_b)
        pred_steps = []
        for idx, ev in enumerate(parsed, 1):
            d = ev.to_dict()
            d["scenario_id"] = scenario["id"]
            d["step"] = idx
            evidence_rows.append(d)
            pred_steps.append({"supports_forward": ev.supports_forward, "strength": ev.strength, "source_weight": ev.source_weight})
        traj = simulate_evidence_steps(pred_steps, learning_rate=args.learning_rate)
        for row in traj:
            row["scenario_id"] = scenario["id"]
            row["scenario_title"] = scenario["title"]
            row["mode"] = "auto"
            trajectory_rows.append(row)

    flat_evidence = []
    for row in evidence_rows:
        meta = row.pop("meta")
        row["matched_rule"] = meta.get("matched_rule", "")
        row["notes"] = "|".join(meta.get("notes", []))
        flat_evidence.append(row)

    with (out / "auto_evidence.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in flat_evidence for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(flat_evidence)
    with (out / "trajectories_auto.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in trajectory_rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(trajectory_rows)
    print(f"Wrote {out / 'auto_evidence.csv'} and {out / 'trajectories_auto.csv'}")

if __name__ == "__main__":
    main()
