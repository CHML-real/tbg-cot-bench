import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUT_DIR = FIGURES / "cumulative_v4_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_metric_file(path):
    metrics = {}
    if not path.exists():
        return metrics
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                metrics[row["metric"]] = float(row["value"])
            except Exception:
                pass
    return metrics


def read_traj(path):
    data = defaultdict(list)
    if not path.exists():
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


def plot_metrics():
    baseline = read_metric_file(RESULTS / "converter_eval_summary.csv")
    stepwise = read_metric_file(RESULTS / "stepwise_ollama_eval_summary.csv")
    order_v3 = read_metric_file(RESULTS / "order_v3_eval_summary.csv")
    cumulative = read_metric_file(RESULTS / "cumulative_v4_eval_summary.csv")

    labels = ["Baseline direction", "Step-wise direction", "Order v3 direction", "Cumulative v4 trajectory"]
    values = [
        baseline.get("direction_accuracy", 0.0),
        stepwise.get("direction_accuracy", 0.0),
        order_v3.get("direction_accuracy", 0.0),
        cumulative.get("trajectory_verdict_accuracy", 0.0),
    ]

    plt.figure(figsize=(11, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Extraction / trajectory accuracy comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out = FIGURES / "cumulative_v4_accuracy_comparison.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved: {out}")


def plot_scenario_comparisons():
    gold = read_traj(RESULTS / "trajectories_gold.csv")
    baseline = read_traj(RESULTS / "trajectories_auto.csv")
    stepwise = read_traj(RESULTS / "trajectories_stepwise_ollama.csv")
    order_v3 = read_traj(RESULTS / "trajectories_order_v3_ollama.csv")
    cumulative = read_traj(RESULTS / "trajectories_cumulative_v4.csv")

    scenario_ids = sorted(set(gold) | set(baseline) | set(stepwise) | set(order_v3) | set(cumulative))

    for sid in scenario_ids:
        plt.figure(figsize=(9, 5))

        for label, data, marker in [
            ("Gold", gold, "o"),
            ("Baseline", baseline, "s"),
            ("Step-wise", stepwise, "^"),
            ("Order v3", order_v3, "D"),
            ("Cumulative v4", cumulative, "x"),
        ]:
            if sid in data:
                xs = [r["step"] for r in data[sid]]
                ys = [r["p_forward"] for r in data[sid]]
                plt.plot(xs, ys, marker=marker, label=label)

        plt.axhline(0.65, linestyle="--", linewidth=1)
        plt.axhline(0.35, linestyle="--", linewidth=1)
        plt.ylim(0, 1)
        plt.xlabel("Step")
        plt.ylabel("p_forward")
        plt.title(f"{sid}: cumulative v4 comparison")
        plt.legend()
        plt.tight_layout()

        out = OUT_DIR / f"{sid.lower()}_cumulative_v4.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"Saved: {out}")


def main():
    plot_metrics()
    plot_scenario_comparisons()


if __name__ == "__main__":
    main()
