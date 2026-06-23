import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUT_DIR = FIGURES / "ollama_comparison"
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


def read_summary_metrics(path):
    metrics = {}
    if not path.exists():
        return metrics

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["metric"]] = float(row["value"])

    return metrics


def plot_all_trajectories(gold, baseline, ollama):
    scenario_ids = sorted(set(gold.keys()) | set(baseline.keys()) | set(ollama.keys()))

    plt.figure(figsize=(13, 7))

    for sid in scenario_ids:
        if sid in ollama:
            xs = [r["step"] for r in ollama[sid]]
            ys = [r["p_forward"] for r in ollama[sid]]
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=sid)

    plt.axhline(0.65, linestyle="--", linewidth=1)
    plt.axhline(0.35, linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.xlabel("Step")
    plt.ylabel("p_forward")
    plt.title("EXAONE via Ollama belief trajectories")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    out = FIGURES / "ollama_trajectories.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved: {out}")


def plot_scenario_comparison(gold, baseline, ollama):
    scenario_ids = sorted(set(gold.keys()) | set(baseline.keys()) | set(ollama.keys()))

    for sid in scenario_ids:
        plt.figure(figsize=(9, 5))

        if sid in gold:
            xs = [r["step"] for r in gold[sid]]
            ys = [r["p_forward"] for r in gold[sid]]
            plt.plot(xs, ys, marker="o", label="Gold")

        if sid in baseline:
            xs = [r["step"] for r in baseline[sid]]
            ys = [r["p_forward"] for r in baseline[sid]]
            plt.plot(xs, ys, marker="s", label="Baseline converter")

        if sid in ollama:
            xs = [r["step"] for r in ollama[sid]]
            ys = [r["p_forward"] for r in ollama[sid]]
            plt.plot(xs, ys, marker="^", label="EXAONE via Ollama")

        plt.axhline(0.65, linestyle="--", linewidth=1)
        plt.axhline(0.35, linestyle="--", linewidth=1)
        plt.ylim(0, 1)
        plt.xlabel("Step")
        plt.ylabel("p_forward")
        plt.title(f"{sid}: Gold vs baseline vs EXAONE")
        plt.legend()
        plt.tight_layout()

        out = OUT_DIR / f"{sid.lower()}_gold_baseline_ollama.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"Saved: {out}")


def plot_accuracy_comparison():
    baseline_metrics = read_summary_metrics(RESULTS / "converter_eval_summary.csv")
    ollama_metrics = read_summary_metrics(RESULTS / "ollama_eval_summary.csv")

    labels = ["Baseline converter", "EXAONE via Ollama"]
    direction_values = [
        baseline_metrics.get("direction_accuracy", 0.0),
        ollama_metrics.get("direction_accuracy", 0.0),
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, direction_values)
    plt.ylim(0, 1)
    plt.ylabel("Direction accuracy")
    plt.title("Evidence direction accuracy")
    plt.tight_layout()

    out = FIGURES / "baseline_vs_ollama_direction_accuracy.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved: {out}")


def main():
    gold = read_trajectory(RESULTS / "trajectories_gold.csv")
    baseline = read_trajectory(RESULTS / "trajectories_auto.csv")
    ollama = read_trajectory(RESULTS / "trajectories_ollama.csv")

    plot_all_trajectories(gold, baseline, ollama)
    plot_scenario_comparison(gold, baseline, ollama)
    plot_accuracy_comparison()


if __name__ == "__main__":
    main()
