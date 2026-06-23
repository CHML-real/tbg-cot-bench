import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUT_DIR = FIGURES / "order_v3_comparison"
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


def read_trajectory(path):
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
        data[sid].sort(key=lambda r: r["step"])
    return data


def plot_accuracy():
    baseline = read_metric_file(RESULTS / "converter_eval_summary.csv")
    scenario = read_metric_file(RESULTS / "ollama_eval_summary.csv")
    stepwise = read_metric_file(RESULTS / "stepwise_ollama_eval_summary.csv")
    order_v3 = read_metric_file(RESULTS / "order_v3_eval_summary.csv")

    labels = ["Baseline", "EXAONE scenario", "EXAONE step-wise", "EXAONE order v3"]
    values = [
        baseline.get("direction_accuracy", 0.0),
        scenario.get("direction_accuracy", 0.0),
        stepwise.get("direction_accuracy", 0.0),
        order_v3.get("direction_accuracy", 0.0),
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Direction accuracy")
    plt.title("Direction accuracy across extraction methods")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = FIGURES / "order_v3_accuracy_comparison.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved: {out}")


def plot_parse_success():
    scenario = read_metric_file(RESULTS / "ollama_eval_summary.csv")
    stepwise = read_metric_file(RESULTS / "stepwise_ollama_eval_summary.csv")
    order_v3 = read_metric_file(RESULTS / "order_v3_eval_summary.csv")

    labels = ["EXAONE scenario", "EXAONE step-wise", "EXAONE order v3"]
    values = [
        scenario.get("num_evaluated_steps", 0.0) / 52.0,
        stepwise.get("parse_success_rate", 0.0),
        order_v3.get("parse_success_rate", 0.0),
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Parse success rate")
    plt.title("Structured output parse success")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = FIGURES / "order_v3_parse_success_comparison.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved: {out}")


def plot_scenario_comparison():
    gold = read_trajectory(RESULTS / "trajectories_gold.csv")
    baseline = read_trajectory(RESULTS / "trajectories_auto.csv")
    stepwise = read_trajectory(RESULTS / "trajectories_stepwise_ollama.csv")
    order_v3 = read_trajectory(RESULTS / "trajectories_order_v3_ollama.csv")

    scenario_ids = sorted(set(gold) | set(baseline) | set(stepwise) | set(order_v3))

    for sid in scenario_ids:
        plt.figure(figsize=(9, 5))

        for label, data, marker in [
            ("Gold", gold, "o"),
            ("Baseline", baseline, "s"),
            ("EXAONE step-wise", stepwise, "^"),
            ("EXAONE order v3", order_v3, "D"),
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
        plt.title(f"{sid}: Order v3 comparison")
        plt.legend()
        plt.tight_layout()

        out = OUT_DIR / f"{sid.lower()}_order_v3_comparison.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"Saved: {out}")


def main():
    plot_accuracy()
    plot_parse_success()
    plot_scenario_comparison()


if __name__ == "__main__":
    main()
