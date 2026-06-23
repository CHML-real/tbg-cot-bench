import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
RESULTS = ROOT / "results"

EVIDENCE_IN = RESULTS / "ollama_order_v3_evidence.csv"
SUMMARY_IN = RESULTS / "order_v3_ollama_scenario_summary.csv"
EVAL_OUT = RESULTS / "order_v3_eval.csv"
SUMMARY_OUT = RESULTS / "order_v3_eval_summary.csv"


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
            }

    return gold, expected_verdict


def main():
    if not EVIDENCE_IN.exists():
        raise FileNotFoundError(f"Missing input file: {EVIDENCE_IN}")

    gold, expected_verdict = load_gold()

    rows = []
    all_steps = 0
    parse_ok_count = 0
    parsed_total = 0
    parsed_correct = 0
    conf_errors = []

    with EVIDENCE_IN.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["scenario_id"]
            try:
                step = int(row["step"])
            except Exception:
                continue

            key = (sid, step)
            if key not in gold:
                continue

            all_steps += 1
            parse_ok = str(row.get("parse_ok", "")).lower() == "true"
            if parse_ok:
                parse_ok_count += 1

            pred = parse_bool(row.get("supports_forward", ""))
            gold_val = gold[key]["supports_forward"]
            direction_correct = pred == gold_val if pred is not None else False

            if pred is not None:
                parsed_total += 1
                if direction_correct:
                    parsed_correct += 1

            confidence = float(row["confidence"]) if row.get("confidence") else 0.5
            gold_strength = gold[key]["strength"]
            conf_error = abs(confidence - gold_strength)

            if pred is not None:
                conf_errors.append(conf_error)

            rows.append({
                "scenario_id": sid,
                "step": step,
                "parse_ok": parse_ok,
                "order": row.get("order", ""),
                "gold_supports_forward": gold_val,
                "pred_supports_forward": "" if pred is None else pred,
                "direction_correct": direction_correct,
                "gold_strength": gold_strength,
                "pred_confidence": confidence,
                "confidence_abs_error": round(conf_error, 6),
                "evidence": row.get("evidence", ""),
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

    parse_success_rate = parse_ok_count / all_steps if all_steps else 0.0
    direction_accuracy = parsed_correct / parsed_total if parsed_total else 0.0
    confidence_mae = sum(conf_errors) / len(conf_errors) if conf_errors else 0.0
    verdict_accuracy = verdict_correct / verdict_total if verdict_total else 0.0

    with EVAL_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "scenario_id",
            "step",
            "parse_ok",
            "order",
            "gold_supports_forward",
            "pred_supports_forward",
            "direction_correct",
            "gold_strength",
            "pred_confidence",
            "confidence_abs_error",
            "evidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with SUMMARY_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["parse_success_rate", round(parse_success_rate, 6)])
        writer.writerow(["direction_accuracy", round(direction_accuracy, 6)])
        writer.writerow(["confidence_mae", round(confidence_mae, 6)])
        writer.writerow(["verdict_accuracy", round(verdict_accuracy, 6)])
        writer.writerow(["num_all_steps", all_steps])
        writer.writerow(["num_parsed_steps", parsed_total])
        writer.writerow(["num_evaluated_verdicts", verdict_total])

    print(f"Saved: {EVAL_OUT}")
    print(f"Saved: {SUMMARY_OUT}")
    print(f"parse_success_rate={parse_success_rate:.4f}")
    print(f"direction_accuracy={direction_accuracy:.4f}")
    print(f"confidence_mae={confidence_mae:.4f}")
    print(f"verdict_accuracy={verdict_accuracy:.4f}")


if __name__ == "__main__":
    main()
