import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PRED_IN = RESULTS / "ollama_cumulative_v4.csv"
GOLD_IN = RESULTS / "trajectories_gold.csv"
EVAL_OUT = RESULTS / "cumulative_v4_eval.csv"
SUMMARY_OUT = RESULTS / "cumulative_v4_eval_summary.csv"
TRAJ_OUT = RESULTS / "trajectories_cumulative_v4.csv"
SCENARIO_SUMMARY_OUT = RESULTS / "cumulative_v4_scenario_summary.csv"


def verdict_from_p(p: float):
    if p > 0.65:
        return "forward"
    if p < 0.35:
        return "backward"
    return "ambiguous"


def load_gold():
    gold = {}
    with GOLD_IN.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["scenario_id"]
            step = int(row["step"])
            p = float(row["p_forward"])
            gold[(sid, step)] = {
                "p_forward": p,
                "verdict": verdict_from_p(p),
            }
    return gold


def main():
    if not PRED_IN.exists():
        raise FileNotFoundError(f"Missing input file: {PRED_IN}")
    if not GOLD_IN.exists():
        raise FileNotFoundError(f"Missing input file: {GOLD_IN}")

    gold = load_gold()

    eval_rows = []
    traj_rows = []
    by_scenario = {}

    total = 0
    parsed = 0
    verdict_correct = 0
    p_abs_errors = []

    with PRED_IN.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["scenario_id"]
            step = int(row["step"])
            key = (sid, step)

            if key not in gold:
                continue

            total += 1
            parse_ok = str(row.get("parse_ok", "")).lower() == "true"
            if parse_ok:
                parsed += 1

            pred_p = float(row["p_forward"])
            pred_verdict = row["verdict"]
            gold_p = gold[key]["p_forward"]
            gold_verdict = gold[key]["verdict"]

            correct = pred_verdict == gold_verdict
            if correct:
                verdict_correct += 1

            p_err = abs(pred_p - gold_p)
            p_abs_errors.append(p_err)

            eval_rows.append({
                "scenario_id": sid,
                "step": step,
                "parse_ok": parse_ok,
                "order": row.get("order", ""),
                "pred_p_forward": pred_p,
                "gold_p_forward": gold_p,
                "p_abs_error": round(p_err, 6),
                "pred_verdict": pred_verdict,
                "gold_verdict": gold_verdict,
                "verdict_correct": correct,
            })

            traj_rows.append({
                "scenario_id": sid,
                "step": step,
                "p_forward": pred_p,
                "verdict": pred_verdict,
                "parse_ok": parse_ok,
            })

            by_scenario[sid] = {
                "final_p": pred_p,
                "final_verdict": pred_verdict,
                "last_step": step,
            }

    parse_success_rate = parsed / total if total else 0.0
    verdict_accuracy = verdict_correct / total if total else 0.0
    p_mae = sum(p_abs_errors) / len(p_abs_errors) if p_abs_errors else 0.0

    with EVAL_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "scenario_id",
            "step",
            "parse_ok",
            "order",
            "pred_p_forward",
            "gold_p_forward",
            "p_abs_error",
            "pred_verdict",
            "gold_verdict",
            "verdict_correct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_rows)

    with TRAJ_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["scenario_id", "step", "p_forward", "verdict", "parse_ok"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(traj_rows)

    with SCENARIO_SUMMARY_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["scenario_id", "final_p", "verdict", "last_step"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sid in sorted(by_scenario):
            item = by_scenario[sid]
            writer.writerow({
                "scenario_id": sid,
                "final_p": item["final_p"],
                "verdict": item["final_verdict"],
                "last_step": item["last_step"],
            })

    with SUMMARY_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["parse_success_rate", round(parse_success_rate, 6)])
        writer.writerow(["trajectory_verdict_accuracy", round(verdict_accuracy, 6)])
        writer.writerow(["p_forward_mae", round(p_mae, 6)])
        writer.writerow(["num_steps", total])
        writer.writerow(["num_parsed_steps", parsed])

    print(f"Saved: {EVAL_OUT}")
    print(f"Saved: {SUMMARY_OUT}")
    print(f"parse_success_rate={parse_success_rate:.4f}")
    print(f"trajectory_verdict_accuracy={verdict_accuracy:.4f}")
    print(f"p_forward_mae={p_mae:.4f}")


if __name__ == "__main__":
    main()
