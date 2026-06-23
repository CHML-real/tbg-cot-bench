"""Core utilities for TBG-CoT-Bench.

The benchmark treats JSON scenario labels as gold evidence.  It uses a simple
log-odds update rule compatible with temporal-belief-graph's Bayesian update
intuition, while remaining dependency-light for notebooks and CI.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import math

DEFAULT_LEARNING_RATE = 0.4
INITIAL_P = 0.5


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, p))
    return math.log(p / (1.0 - p))


def verdict_from_p(p: float) -> str:
    if p > 0.65:
        return "forward"
    if p < 0.35:
        return "backward"
    return "ambiguous"


def load_scenarios(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    files = sorted(path.glob("sc*.json")) if path.is_dir() else [path]
    return [json.loads(file.read_text(encoding="utf-8")) for file in files]


def simulate_evidence_steps(
    steps: Iterable[dict[str, Any]],
    *,
    initial_p: float = INITIAL_P,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> list[dict[str, Any]]:
    p = initial_p
    rows: list[dict[str, Any]] = [
        {"step": 0, "p_forward": round(p, 4), "delta": 0.0, "direction_flips": 0, "skipped": False}
    ]
    previous_delta: float | None = None
    flips = 0

    for idx, evidence in enumerate(steps, start=1):
        support = evidence.get("supports_forward")
        if support is None:
            rows.append({
                "step": idx,
                "p_forward": round(p, 4),
                "delta": 0.0,
                "direction_flips": flips,
                "skipped": True,
                "note": "ambiguous_direction",
            })
            continue

        direction = 1 if support else -1
        old_p = p
        p = sigmoid(logit(p) + direction * float(evidence["strength"]) * float(evidence["source_weight"]) * learning_rate)
        p = max(0.001, min(0.999, p))
        delta = p - old_p

        if previous_delta is not None and delta * previous_delta < 0:
            flips += 1
        if abs(delta) > 1e-12:
            previous_delta = delta

        rows.append({
            "step": idx,
            "p_forward": round(p, 4),
            "delta": round(delta, 4),
            "direction_flips": flips,
            "skipped": False,
        })

    return rows


def simulate_gold(
    scenario: dict[str, Any],
    *,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> list[dict[str, Any]]:
    rows = simulate_evidence_steps(scenario["steps"], learning_rate=learning_rate)
    for row in rows:
        row["scenario_id"] = scenario["id"]
        row["scenario_title"] = scenario["title"]
        row["mode"] = "gold"
    return rows


def summarize_trajectory(scenario: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    final_p = rows[-1]["p_forward"]
    return {
        "scenario_id": scenario["id"],
        "title": scenario["title"],
        "type": scenario["type"],
        "pattern": scenario["pattern"],
        "num_steps": len(scenario["steps"]),
        "final_p": final_p,
        "verdict": verdict_from_p(final_p),
        "direction_flips": rows[-1]["direction_flips"],
        "final_conviction": round(abs(final_p - 0.5), 4),
        "expected_validator_flags": "|".join(scenario.get("expected_validator_flags", [])),
    }
