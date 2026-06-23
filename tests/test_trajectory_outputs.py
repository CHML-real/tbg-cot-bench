import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_trajectory(name):
    path = RESULTS / name
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def test_gold_trajectory_has_all_scenarios():
    rows = load_trajectory("trajectories_gold.csv")
    scenarios = {row["scenario_id"] for row in rows}
    assert len(scenarios) == 10


def test_stepwise_trajectory_probability_bounds():
    rows = load_trajectory("trajectories_stepwise_ollama.csv")
    assert rows, "trajectories_stepwise_ollama.csv is empty"

    for row in rows:
        p = float(row["p_forward"])
        assert 0.0 <= p <= 1.0


def test_stepwise_trajectory_has_initial_state_per_scenario():
    rows = load_trajectory("trajectories_stepwise_ollama.csv")
    by_scenario = defaultdict(list)

    for row in rows:
        by_scenario[row["scenario_id"]].append(row)

    assert len(by_scenario) == 10

    for sid, items in by_scenario.items():
        has_initial = any(int(row["step"]) == 0 and abs(float(row["p_forward"]) - 0.5) < 1e-9 for row in items)
        assert has_initial, f"{sid} missing initial p_forward=0.5 state"
