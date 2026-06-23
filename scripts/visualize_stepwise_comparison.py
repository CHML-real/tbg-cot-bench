import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUT_DIR = FIGURES / "stepwise_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_trajectory(path):
    data = defaultdict(list)
    if not path.exists():
        print(f"Missing: {path}")
        return data

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["scenario_id"]].append({
                "step": int(row["step"]),
                "p_forward": float(row["p_forward"]),
            })

    for sid in data:
        data[sid].sort(key=lambda x: x["step"])

    return data


def read_metric_file(path):
    metrics = {}
    if not path.exists():
        return metrics
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                metrics[row["metric"]] = float(row["value"])
            except (ValueError, KeyError):
                pass
    return metrics


def plot_accuracy_comparison():
    baseline = read_metric_file(RESULTS / "converter_eval_summary.csv")
    scenario_level = read_metric_file(RESULTS / "ollama_eval_summary.csv")
    stepwise = read_metric_file(RESULTS / "stepwise_ollama_eval_summary.csv")

    labels = ["Baseline", "EXAONE scenario", "EXAONE step-wise"]
    values = [
        baseline.get("direction_accuracy", 0.0),
        scenario_level.get("direction_accuracy", 0.0),
        stepwise.get("direction_accuracy", 0.0),
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Direction accuracy")
    plt.title("Direction accuracy comparison")
    plt.tight_layout()
    out = FIGURES / "stepwise_vs_baseline_accuracy.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved: {out}")


def plot_parse_success():
    scenario_level = read_metric_file(RESULTS / "ollama_eval_summary.csv")
    stepwise = read_metric_file(RESULTS / "stepwise_ollama_eval_summary.csv")

    # Scenario-level parse success can be inferred from evaluated steps / 52 if present.
    scenario_steps = scenario_level.get("num_evaluated_steps", 0.0)
    scenario_parse_rate = scenario_steps / 52.0 if scenario_steps else 0.0

    labels = ["EXAONE scenario", "EXAONE step-wise"]
    values = [
        scenario_parse_rate,
        stepwise.get("parse_success_rate", 0.0),
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Parse success rate")
    plt.title("Structured output parse success")
    plt.tight_layout()
    out = FIGURES / "stepwise_parse_success.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved: {out}")


def plot_all_stepwise_trajectories(stepwise):
    plt.figure(figsize=(13, 7))

    for sid in sorted(stepwise.keys()):
        xs = [r["step"] for r in stepwise[sid]]
        ys = [r["p_forward"] for r in stepwise[sid]]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=sid)

    plt.axhline(0.65, linestyle="--", linewidth=1)
    plt.axhline(0.35, linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.xlabel("Step")
    plt.ylabel("p_forward")
    plt.title("EXAONE step-wise belief trajectories")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    out = FIGURES / "stepwise_ollama_trajectories.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved: {out}")


def plot_scenario_comparisons(gold, baseline, scenario_ollama, stepwise):
    scenario_ids = sorted(
        set(gold.keys())
        | set(baseline.keys())
        | set(scenario_ollama.keys())
        | set(stepwise.keys())
    )

    for sid in scenario_ids:
        plt.figure(figsize=(9, 5))

        if sid in gold:
            xs = [r["step"] for r in gold[sid]]
            ys = [r["p_forward"] for r in gold[sid]]
            plt.plot(xs, ys, marker="o", label="Gold")

        if sid in baseline:
            xs = [r["step"] for r in baseline[sid]]
            ys = [r["p_forward"] for r in baseline[sid]]
            plt.plot(xs, ys, marker="s", label="Baseline")

        if sid in scenario_ollama:
            xs = [r["step"] for r in scenario_ollama[sid]]
            ys = [r["p_forward"] for r in scenario_ollama[sid]]
            plt.plot(xs, ys, marker="^", label="EXAONE scenario")

        if sid in stepwise:
            xs = [r["step"] for r in stepwise[sid]]
            ys = [r["p_forward"] for r in stepwise[sid]]
            plt.plot(xs, ys, marker="D", label="EXAONE step-wise")

        plt.axhline(0.65, linestyle="--", linewidth=1)
        plt.axhline(0.35, linestyle="--", linewidth=1)
        plt.ylim(0, 1)
        plt.xlabel("Step")
        plt.ylabel("p_forward")
        plt.title(f"{sid}: Gold vs baseline vs EXAONE variants")
        plt.legend()
        plt.tight_layout()

        out = OUT_DIR / f"{sid.lower()}_comparison.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"Saved: {out}")


def main():
    gold = read_trajectory(RESULTS / "trajectories_gold.csv")
    baseline = read_trajectory(RESULTS / "trajectories_auto.csv")
    scenario_ollama = read_trajectory(RESULTS / "trajectories_ollama.csv")
    stepwise = read_trajectory(RESULTS / "trajectories_stepwise_ollama.csv")

    plot_accuracy_comparison()
    plot_parse_success()
    plot_all_stepwise_trajectories(stepwise)
    plot_scenario_comparisons(gold, baseline, scenario_ollama, stepwise)


if __name__ == "__main__":
    main()
