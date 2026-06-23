import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"

REPORTS.mkdir(parents=True, exist_ok=True)
OUT = REPORTS / "tbg_cot_experiment_report.md"


def read_metric_file(path: Path) -> dict:
    metrics = {}
    if not path.exists():
        return metrics

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                metrics[row["metric"]] = float(row["value"])
            except Exception:
                continue
    return metrics


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def method_rows():
    baseline = read_metric_file(RESULTS / "converter_eval_summary.csv")
    scenario = read_metric_file(RESULTS / "ollama_eval_summary.csv")
    stepwise = read_metric_file(RESULTS / "stepwise_ollama_eval_summary.csv")
    order_v3 = read_metric_file(RESULTS / "order_v3_eval_summary.csv")
    cumulative_v4 = read_metric_file(RESULTS / "cumulative_v4_eval_summary.csv")

    return [
        {
            "method": "Rule-based baseline converter",
            "parse_success": 1.0 if baseline else None,
            "primary_metric": "direction_accuracy",
            "primary_value": baseline.get("direction_accuracy"),
            "secondary": f"strength_mae={fmt(baseline.get('strength_mae'))}, source_weight_mae={fmt(baseline.get('source_weight_mae'))}",
            "coverage": f"{int(baseline.get('num_steps', 0))}/52",
        },
        {
            "method": "EXAONE scenario-level CoT",
            "parse_success": scenario.get("num_evaluated_steps", 0) / 52.0 if scenario else None,
            "primary_metric": "direction_accuracy",
            "primary_value": scenario.get("direction_accuracy"),
            "secondary": f"confidence_mae={fmt(scenario.get('confidence_mae'))}, verdict_accuracy={fmt(scenario.get('verdict_accuracy'))}",
            "coverage": f"{int(scenario.get('num_evaluated_steps', 0))}/52",
        },
        {
            "method": "EXAONE step-wise evidence v2.1",
            "parse_success": stepwise.get("parse_success_rate"),
            "primary_metric": "direction_accuracy",
            "primary_value": stepwise.get("direction_accuracy"),
            "secondary": f"confidence_mae={fmt(stepwise.get('confidence_mae'))}, verdict_accuracy={fmt(stepwise.get('verdict_accuracy'))}",
            "coverage": f"{int(stepwise.get('num_parsed_steps', 0))}/52",
        },
        {
            "method": "EXAONE order-classification v3",
            "parse_success": order_v3.get("parse_success_rate"),
            "primary_metric": "direction_accuracy",
            "primary_value": order_v3.get("direction_accuracy"),
            "secondary": f"confidence_mae={fmt(order_v3.get('confidence_mae'))}, verdict_accuracy={fmt(order_v3.get('verdict_accuracy'))}",
            "coverage": f"{int(order_v3.get('num_parsed_steps', 0))}/52",
        },
        {
            "method": "EXAONE cumulative belief v4",
            "parse_success": cumulative_v4.get("parse_success_rate"),
            "primary_metric": "trajectory_verdict_accuracy",
            "primary_value": cumulative_v4.get("trajectory_verdict_accuracy"),
            "secondary": f"p_forward_mae={fmt(cumulative_v4.get('p_forward_mae'))}",
            "coverage": f"{int(cumulative_v4.get('num_parsed_steps', 0))}/52",
        },
    ]


def render_method_table() -> str:
    lines = [
        "| Method | Parse success | Primary metric | Primary value | Secondary metrics | Coverage |",
        "|---|---:|---|---:|---|---:|",
    ]

    for row in method_rows():
        lines.append(
            f"| {row['method']} | {fmt(row['parse_success'])} | {row['primary_metric']} | "
            f"{fmt(row['primary_value'])} | {row['secondary']} | {row['coverage']} |"
        )

    return "\n".join(lines)


def render_scenario_summary(path: Path, title: str) -> str:
    rows = read_rows(path)
    if not rows:
        return f"## {title}\n\nNo data found at `{path.relative_to(ROOT)}`.\n"

    lines = [
        f"## {title}",
        "",
        "| Scenario | Final p_forward | Verdict | Parse / coverage |",
        "|---|---:|---|---|",
    ]

    for row in rows:
        sid = row.get("scenario_id", "")
        final_p = row.get("final_p", "")
        verdict = row.get("verdict", "")
        parsed = row.get("parsed_steps", "")
        total = row.get("total_steps", "")
        rate = row.get("parse_success_rate", "")

        coverage = ""
        if parsed or total or rate:
            coverage = f"{parsed}/{total}, rate={rate}"

        lines.append(f"| {sid} | {fmt(final_p)} | {verdict} | {coverage} |")

    return "\n".join(lines)


def render_assets() -> str:
    candidates = [
        "figures/converter_direction_accuracy.png",
        "figures/gold_trajectories.png",
        "figures/stepwise_vs_baseline_accuracy.png",
        "figures/stepwise_parse_success.png",
        "figures/order_v3_accuracy_comparison.png",
        "figures/order_v3_parse_success_comparison.png",
        "figures/cumulative_v4_accuracy_comparison.png",
    ]

    lines = ["## Generated Assets", ""]
    for rel in candidates:
        path = ROOT / rel
        status = "available" if path.exists() else "missing"
        lines.append(f"- `{rel}` — {status}")
    return "\n".join(lines)


def main():
    baseline = read_metric_file(RESULTS / "converter_eval_summary.csv")
    stepwise = read_metric_file(RESULTS / "stepwise_ollama_eval_summary.csv")
    cumulative = read_metric_file(RESULTS / "cumulative_v4_eval_summary.csv")

    baseline_acc = baseline.get("direction_accuracy")
    stepwise_acc = stepwise.get("direction_accuracy")
    stepwise_parse = stepwise.get("parse_success_rate")
    cumulative_parse = cumulative.get("parse_success_rate")
    cumulative_acc = cumulative.get("trajectory_verdict_accuracy")

    report = f"""# TBG-CoT-Bench Local Experiment Report

Generated at: `{datetime.now().isoformat(timespec="seconds")}`

## Overview

This report summarizes a local application benchmark for temporal belief tracking.

The benchmark tests whether a system can track evidence about the temporal claim:

> Event A occurred before Event B.

The current project should be interpreted as an **application-level benchmark / usage test**, not as a full internal unit test suite for the upstream `temporal-belief-graph` package.

## Compared Methods

{render_method_table()}

## Main Finding

The strongest current method is:

**EXAONE step-wise evidence v2.1**

It achieved:

- parse success rate: `{fmt(stepwise_parse)}`
- direction accuracy: `{fmt(stepwise_acc)}`
- rule-based baseline direction accuracy: `{fmt(baseline_acc)}`

This suggests that local EXAONE becomes useful when the task is decomposed into small structured evidence judgments.

## Negative Result

The cumulative v4 method was conceptually closer to belief trajectory tracking, but performed worse in this run:

- cumulative v4 parse success rate: `{fmt(cumulative_parse)}`
- cumulative v4 trajectory verdict accuracy: `{fmt(cumulative_acc)}`

This indicates that cumulative prompting can overload the local model and reduce structured-output stability.

## Interpretation

The experiment supports the following conclusion:

1. Scenario-level CoT is unstable for local EXAONE in this setup.
2. Step-wise extraction substantially improves structured-output reliability.
3. Order classification alone tends to create `UNCLEAR` collapse.
4. Cumulative belief prompting is not yet reliable with this local model/configuration.
5. The best current architecture is a modular pipeline:
   - evidence extraction
   - structured parsing
   - trajectory update
   - result visualization

{render_scenario_summary(RESULTS / "stepwise_ollama_scenario_summary.csv", "Step-wise v2.1 Scenario Summary")}

{render_scenario_summary(RESULTS / "cumulative_v4_scenario_summary.csv", "Cumulative v4 Scenario Summary")}

{render_assets()}

## Recommended Next Step

Freeze **EXAONE step-wise evidence v2.1** as the current best local method.

Next development should focus on:

- application-level regression tests
- reproducible reports
- HuggingFace notebook packaging
- optional integration tests against the actual `temporal-belief-graph` package

"""

    OUT.write_text(report, encoding="utf-8")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
