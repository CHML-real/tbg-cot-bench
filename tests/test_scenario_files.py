import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"


def test_scenario_directory_exists():
    assert SCENARIOS.exists(), "scenarios/ directory is missing"


def test_scenario_count():
    files = sorted(SCENARIOS.glob("sc*.json"))
    assert len(files) == 10, f"Expected 10 scenario files, found {len(files)}"


def test_scenario_schema_minimum():
    required_top = {"id", "events", "steps", "expected_verdict"}

    for path in sorted(SCENARIOS.glob("sc*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))

        assert required_top.issubset(data.keys()), f"{path.name} missing required top-level fields"

        assert "event_a" in data["events"], f"{path.name} missing events.event_a"
        assert "event_b" in data["events"], f"{path.name} missing events.event_b"
        assert "label" in data["events"]["event_a"], f"{path.name} missing event_a.label"
        assert "label" in data["events"]["event_b"], f"{path.name} missing event_b.label"

        assert isinstance(data["steps"], list), f"{path.name} steps must be a list"
        assert len(data["steps"]) >= 4, f"{path.name} should have at least 4 steps"

        for i, step in enumerate(data["steps"], start=1):
            for key in ["text", "supports_forward", "strength", "source_weight"]:
                assert key in step, f"{path.name} step {i} missing {key}"
            assert isinstance(step["supports_forward"], bool), f"{path.name} step {i} supports_forward must be bool"
            assert 0.0 <= float(step["strength"]) <= 1.0, f"{path.name} step {i} strength out of range"
            assert 0.0 < float(step["source_weight"]) <= 2.0, f"{path.name} step {i} source_weight out of expected range"
