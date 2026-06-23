import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
RESULTS = ROOT / "results"

EVIDENCE_IN = RESULTS / "ollama_evidence.csv"
SUMMARY_IN = RESULTS / "ollama_scenario_summary.csv"
EVAL_OUT = RESULTS / "ollama_eval.csv"
SUMMARY_OUT = RESULTS / "ollama_eval_summary.csv"


def parse_bool(value):
    s = str(value).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None


def load_gold():
    gold = {}
    expected_verdict = {}

    for path in sorted(SCENARIOS.glob("sc*.json")):
        scenario = json.loads(path.read_text(encoding="utf-8"))
        sid = scenario["id"]
        expected_verdict[sid] = scenario.get("expected_verdict", "")

        for i, step in enumerate(scenario["steps"], start=1):
            gold[(sid, i)] = {
                "supports_forward": bool(step["supports_forward"]),
                "strength": float(step["strength"]),
                "source_weight": float(step["source_weight"]),
            }

    return gold, expected_verdict


def main():
    if not EVIDENCE_IN.exists():
        raise FileNotFoundError(f"Missing input file: {EVIDENCE_IN}")

    gold, expected_verdict = load_gold()

    eval_rows = []
    correct = 0
    total = 0
    confidence_abs_errors = []

    with EVIDENCE_IN.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("parse_ok", "")).lower() != "true":
                continue

            try:
                step = int(row["step"])
            except (TypeError, ValueError):
                continue

            sid = row["scenario_id"]
            key = (sid, step)
            if key not in gold:
                continue

            pred = parse_bool(row["supports_forward"])
            gold_val = gold[key]["supports_forward"]
            direction_correct = pred == gold_val

            if pred is not None:
                total += 1
                if direction_correct:
                    correct += 1

            confidence = float(row["confidence"]) if row["confidence"] else 0.5
            gold_strength = gold[key]["strength"]
            conf_err = abs(confidence - gold_strength)
            confidence_abs_errors.append(conf_err)

            eval_rows.append({
                "scenario_id": sid,
                "step": step,
                "gold_supports_forward": gold_val,
                "pred_supports_forward": "" if pred is None else pred,
                "direction_correct": direction_correct,
                "gold_strength": gold_strength,
                "pred_confidence": confidence,
                "confidence_abs_error": round(conf_err, 6),
                "text": row.get("text", ""),
            })

    scenario_verdicts = {}
    if SUMMARY_IN.exists():
        with SUMMARY_IN.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenario_verdicts[row["scenario_id"]] = row.get("verdict", "")

    verdict_total = 0
    verdict_correct = 0
    for sid, expected in expected_verdict.items():
        pred = scenario_verdicts.get(sid)
        if pred:
            verdict_total += 1
            if pred == expected:
                verdict_correct += 1

    direction_accuracy = correct / total if total else 0.0
    confidence_mae = (
        sum(confidence_abs_errors) / len(confidence_abs_errors)
        if confidence_abs_errors else 0.0
    )
    verdict_accuracy = verdict_correct / verdict_total if verdict_total else 0.0

    RESULTS.mkdir(exist_ok=True)

    with EVAL_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "scenario_id",
            "step",
            "gold_supports_forward",
            "pred_supports_forward",
            "direction_correct",
            "gold_strength",
            "pred_confidence",
            "confidence_abs_error",
            "text",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_rows)

    with SUMMARY_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["direction_accuracy", round(direction_accuracy, 6)])
        writer.writerow(["confidence_mae", round(confidence_mae, 6)])
        writer.writerow(["verdict_accuracy", round(verdict_accuracy, 6)])
        writer.writerow(["num_evaluated_steps", total])
        writer.writerow(["num_evaluated_verdicts", verdict_total])

    print(f"Saved: {EVAL_OUT}")
    print(f"Saved: {SUMMARY_OUT}")
    print(f"direction_accuracy={direction_accuracy:.4f}")
    print(f"confidence_mae={confidence_mae:.4f}")
    print(f"verdict_accuracy={verdict_accuracy:.4f}")


if __name__ == "__main__":
    main()
