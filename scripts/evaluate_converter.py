#!/usr/bin/env python
from __future__ import annotations
import argparse, csv
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from statistics import mean
from tbg_cot_bench.core import load_scenarios
from tbg_cot_bench.converter_baseline import EvidenceConverter


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline converter against gold scenario labels.")
    parser.add_argument("--scenarios", default="scenarios")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    converter = EvidenceConverter()
    rows = []
    for scenario in load_scenarios(args.scenarios):
        event_a = scenario["events"]["event_a"]["label"]
        event_b = scenario["events"]["event_b"]["label"]
        parsed = converter.convert_steps([s["text"] for s in scenario["steps"]], scenario_id=scenario["id"], event_a_label=event_a, event_b_label=event_b)
        for idx, (gold, pred) in enumerate(zip(scenario["steps"], parsed), start=1):
            pred_dict = pred.to_dict()
            rows.append({
                "scenario_id": scenario["id"],
                "step": idx,
                "gold_supports_forward": gold["supports_forward"],
                "pred_supports_forward": pred.supports_forward,
                "direction_correct": pred.supports_forward == gold["supports_forward"],
                "gold_strength": gold["strength"],
                "pred_strength": pred.strength,
                "strength_abs_error": round(abs(float(gold["strength"]) - pred.strength), 4),
                "gold_source": gold["source"],
                "pred_source": pred.source_label,
                "gold_source_weight": gold["source_weight"],
                "pred_source_weight": pred.source_weight,
                "source_weight_abs_error": round(abs(float(gold["source_weight"]) - pred.source_weight), 4),
                "matched_rule": pred_dict["meta"]["matched_rule"],
                "notes": "|".join(pred_dict["meta"]["notes"]),
            })

    fieldnames = list(rows[0].keys())
    with (out / "converter_eval.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)

    acc = mean(1.0 if r["direction_correct"] else 0.0 for r in rows)
    strength_mae = mean(float(r["strength_abs_error"]) for r in rows)
    weight_mae = mean(float(r["source_weight_abs_error"]) for r in rows)
    summary = [
        {"metric": "direction_accuracy", "value": round(acc, 4)},
        {"metric": "strength_mae", "value": round(strength_mae, 4)},
        {"metric": "source_weight_mae", "value": round(weight_mae, 4)},
        {"metric": "num_steps", "value": len(rows)},
    ]
    with (out / "converter_eval_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader(); writer.writerows(summary)
    print(f"Direction accuracy: {acc:.3f}; wrote {out / 'converter_eval.csv'}")

if __name__ == "__main__":
    main()
