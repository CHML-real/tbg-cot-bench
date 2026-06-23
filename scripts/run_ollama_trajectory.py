import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
RESULTS = ROOT / "results"

IN_PATH = RESULTS / "ollama_evidence.csv"
TRAJ_OUT = RESULTS / "trajectories_ollama.csv"
SUMMARY_OUT = RESULTS / "ollama_scenario_summary.csv"

LEARNING_RATE = 0.4


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def verdict_from_p(p: float) -> str:
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
            if str(row.get("parse_ok", "")).lower() != "true":
                continue
            try:
                step = int(row["step"])
            except (TypeError, ValueError):
                continue
            by_scenario[row["scenario_id"]].append(row)

    trajectory_rows = []
    summary_rows = []

    for sid in sorted(by_scenario.keys()):
        rows = sorted(by_scenario[sid], key=lambda r: int(r["step"]))

        p = 0.5
        prev_delta = None
        flips = 0

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
            step = int(row["step"])
            supports_forward = parse_bool(row["supports_forward"])
            confidence = float(row["confidence"]) if row["confidence"] else 0.5
            source_weight = source_weights.get((sid, step), 1.0)

            if supports_forward is None:
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

            direction = 1 if supports_forward else -1
            old_p = p
            old_logit = math.log(p / (1 - p))
            new_logit = old_logit + direction * confidence * source_weight * LEARNING_RATE
            p = max(0.001, min(0.999, sigmoid(new_logit)))

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

        summary_rows.append({
            "scenario_id": sid,
            "final_p": round(p, 6),
            "verdict": verdict_from_p(p),
            "direction_flips": flips,
            "num_steps": len(rows),
        })

    RESULTS.mkdir(exist_ok=True)

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
        fieldnames = ["scenario_id", "final_p", "verdict", "direction_flips", "num_steps"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved: {TRAJ_OUT}")
    print(f"Saved: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
