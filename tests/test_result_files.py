from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


REQUIRED_RESULT_FILES = [
    "converter_eval_summary.csv",
    "converter_eval.csv",
    "trajectories_gold.csv",
    "trajectories_auto.csv",
    "stepwise_ollama_eval_summary.csv",
    "stepwise_ollama_eval.csv",
    "trajectories_stepwise_ollama.csv",
]


def test_results_directory_exists():
    assert RESULTS.exists(), "results/ directory is missing"


def test_required_result_files_exist():
    missing = [name for name in REQUIRED_RESULT_FILES if not (RESULTS / name).exists()]
    assert not missing, f"Missing result files: {missing}"


def test_summary_files_have_metric_value_columns():
    summary_files = [
        RESULTS / "converter_eval_summary.csv",
        RESULTS / "stepwise_ollama_eval_summary.csv",
    ]

    for path in summary_files:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["metric", "value"], f"{path.name} must have metric,value columns"
            rows = list(reader)
            assert rows, f"{path.name} must not be empty"
