import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
RESULTS = ROOT / "results"

IN_PATH = RESULTS / "ollama_order_v3_evidence.csv"
TRAJ_OUT = RESULTS / "trajectories_order_v3_ollama.csv"
SUMMARY_OUT = RESULTS / "order_v3_ollama_scenario_summary.csv"

LEARNING_RATE = 0.4


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def verdict_from_p(p):
    if p > 0.65:
        return "forward"
    if p < 0.35:
        return "backward"
    return "ambiguous"


def parse_bool(value):
    s = str(value).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None


def load_source_weights():
    weights = {}
    for path in sorted(SCENARIOS.glob("sc*.json")):
        scenario = json.loads(path.read_text(encoding="utf-8"))
        sid = scenario["id"]
        for i, step in enumerate(scenario["steps"], start=1):
            weights[(sid, i)] = float(step.get("source_weight", 1.0))
    return weights


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {IN_PATH}")

    source_weights = load_source_weights()
    by_scenario = defaultdict(list)

    with IN_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["step_int"] = int(row["step"])
            except Exception:
                continue
            by_scenario[row["scenario_id"]].append(row)

    trajectory_rows = []
    summary_rows = []

    for sid in sorted(by_scenario.keys()):
        rows = sorted(by_scenario[sid], key=lambda r: r["step_int"])

        p = 0.5
        prev_delta = None
        flips = 0
        parsed_steps = 0
        skipped_steps = 0

        trajectory_rows.append({
            "scenario_id": sid,
            "step": 0,
            "p_forward": round(p, 6),
            "delta": 0.0,
            "direction_flips": 0,
            "supports_forward": "",
            "confidence": "",
            "source_weight": "",
            "skipped": False,
        })

        for row in rows:
            step = row["step_int"]
            parse_ok = str(row.get("parse_ok", "")).lower() == "true"
            supports_forward = parse_bool(row.get("supports_forward", ""))
            confidence = float(row["confidence"]) if row.get("confidence") else 0.5
            source_weight = source_weights.get((sid, step), 1.0)

            if not parse_ok or supports_forward is None:
                skipped_steps += 1
                trajectory_rows.append({
                    "scenario_id": sid,
                    "step": step,
                    "p_forward": round(p, 6),
                    "delta": 0.0,
                    "direction_flips": flips,
                    "supports_forward": "",
                    "confidence": confidence,
                    "source_weight": source_weight,
                    "skipped": True,
                })
                continue

            parsed_steps += 1
            direction = 1 if supports_forward else -1
            old_p = p
            old_logit = math.log(p / (1 - p))
            p = sigmoid(old_logit + direction * confidence * source_weight * LEARNING_RATE)
            p = max(0.001, min(0.999, p))

            delta = p - old_p
            if prev_delta is not None and delta * prev_delta < 0:
                flips += 1
            if delta != 0:
                prev_delta = delta

            trajectory_rows.append({
                "scenario_id": sid,
                "step": step,
                "p_forward": round(p, 6),
                "delta": round(delta, 6),
                "direction_flips": flips,
                "supports_forward": supports_forward,
                "confidence": confidence,
                "source_weight": source_weight,
                "skipped": False,
            })

        total_steps = parsed_steps + skipped_steps

        summary_rows.append({
            "scenario_id": sid,
            "final_p": round(p, 6),
            "verdict": verdict_from_p(p),
            "direction_flips": flips,
            "parsed_steps": parsed_steps,
            "skipped_steps": skipped_steps,
            "total_steps": total_steps,
            "parse_success_rate": round(parsed_steps / total_steps, 6) if total_steps else 0.0,
        })

    with TRAJ_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "scenario_id",
            "step",
            "p_forward",
            "delta",
            "direction_flips",
            "supports_forward",
            "confidence",
            "source_weight",
            "skipped",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trajectory_rows)

    with SUMMARY_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "scenario_id",
            "final_p",
            "verdict",
            "direction_flips",
            "parsed_steps",
            "skipped_steps",
            "total_steps",
            "parse_success_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved: {TRAJ_OUT}")
    print(f"Saved: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
