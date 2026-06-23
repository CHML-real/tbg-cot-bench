import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_metrics(name):
    path = RESULTS / name
    metrics = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["metric"]] = float(row["value"])
    return metrics


def test_stepwise_parse_success_is_high():
    metrics = read_metrics("stepwise_ollama_eval_summary.csv")
    assert metrics["parse_success_rate"] >= 0.90


def test_stepwise_beats_rule_based_baseline_on_direction_accuracy():
    baseline = read_metrics("converter_eval_summary.csv")
    stepwise = read_metrics("stepwise_ollama_eval_summary.csv")

    assert stepwise["direction_accuracy"] > baseline["direction_accuracy"]


def test_stepwise_has_near_full_step_coverage():
    metrics = read_metrics("stepwise_ollama_eval_summary.csv")
    assert metrics["num_all_steps"] == 52
    assert metrics["num_parsed_steps"] >= 48
