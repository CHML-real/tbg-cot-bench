# TBG-CoT-Bench

**TBG-CoT-Bench** is a local application benchmark for testing temporal belief tracking over Chain-of-Thought-style evidence sequences.

The benchmark evaluates whether a system can track belief about the temporal claim:

> Event A occurred before Event B.

This repository contains synthetic temporal reasoning scenarios, rule-based baselines, local EXAONE/Ollama experiments, trajectory visualizations, generated reports, and pytest-based application benchmark checks.

---

## Current Status

```text
Application benchmark tests: 13 passed
Best current method: EXAONE step-wise evidence v2.1
```

The current best local method is **EXAONE step-wise evidence extraction v2.1**, which achieved stronger direction accuracy than the rule-based converter baseline while maintaining high structured-output parse stability.

---

## What This Repository Tests

This repository is an **application-level benchmark / usage test**, not a full internal unit test suite for the upstream `temporal-belief-graph` package.

It tests the following workflow:

```text
scenario JSON
→ evidence extraction
→ structured parsing
→ temporal belief trajectory update
→ evaluation
→ visualization
→ report generation
→ pytest validation
```

The goal is to check whether temporal evidence can be converted into a belief trajectory over the probability that Event A occurred before Event B.

---

## Compared Methods

The benchmark currently compares:

1. **Rule-based baseline converter**
   A deterministic evidence converter using simple extraction rules.

2. **EXAONE scenario-level CoT**
   The model receives a full scenario and returns all step judgments at once.

3. **EXAONE step-wise evidence v2.1**
   The model receives one evidence item at a time and returns a structured judgment.

4. **EXAONE order-classification v3**
   The model chooses between `A_BEFORE_B`, `B_BEFORE_A`, and `UNCLEAR`.

5. **EXAONE cumulative belief v4**
   The model receives cumulative evidence and directly estimates the current temporal conclusion.

---

## Main Finding

The strongest current method is:

```text
EXAONE step-wise evidence v2.1
```

This method performed best because it decomposes the task into small structured judgments. In this local run, it outperformed the rule-based baseline while avoiding the JSON instability seen in scenario-level prompting.

The cumulative v4 method was conceptually closer to belief trajectory tracking, but it reduced parse stability and produced weaker trajectory agreement in the current local setup.

---

## Repository Structure

```text
tbg-cot-bench-local-experiment/
├── scenarios/              # Temporal reasoning benchmark scenarios
├── scripts/                # Experiment, evaluation, visualization, and report scripts
├── results/                # CSV and JSONL experiment outputs
├── figures/                # Generated plots and comparison figures
├── reports/                # Markdown experiment reports
├── tests/                  # Application-level benchmark tests
├── pytest.ini
├── README.md
└── DATASET_CARD.md
```

---

## Requirements

This project was tested locally with:

```text
Ubuntu 22.04
Python virtual environment
Ollama
EXAONE local GGUF model via Ollama
pytest
matplotlib
```

Install basic Python dependencies:

```bash
pip install pytest matplotlib
```

Ollama should already have a local model registered, for example:

```bash
ollama list
```

Expected model name used in scripts:

```text
exaone-local
```

---

## Running the Benchmark

Activate the virtual environment from the project root:

```bash
cd ~/Desktop/CHMLabs/tbg-cot-bench-local-experiment
source ../venv/bin/activate
```

Run the current best EXAONE step-wise pipeline:

```bash
OLLAMA_MODEL=exaone-local bash scripts/run_stepwise_v21_pipeline.sh
```

Generate the experiment report:

```bash
python scripts/generate_experiment_report.py
```

Run application benchmark tests:

```bash
pytest
```

Expected result:

```text
13 passed
```

---

## Key Result Files

```text
results/converter_eval_summary.csv
results/stepwise_ollama_eval_summary.csv
results/order_v3_eval_summary.csv
results/cumulative_v4_eval_summary.csv
reports/tbg_cot_experiment_report.md
```

Important visualizations include:

```text
figures/stepwise_vs_baseline_accuracy.png
figures/stepwise_parse_success.png
figures/order_v3_accuracy_comparison.png
figures/cumulative_v4_accuracy_comparison.png
```

---

## Testing

The test suite validates the benchmark as a reproducible application-level experiment.

It checks:

```text
- scenario file structure
- required result files
- step-wise v2.1 performance against the baseline
- trajectory probability bounds
- report generation
```

Run:

```bash
pytest
```

Current validated result:

```text
13 passed
```

---

## Interpretation

The experiments suggest the following:

1. Scenario-level CoT prompting is unstable for local EXAONE in this setup.
2. Step-wise evidence extraction improves structured-output reliability.
3. Order classification can collapse into `UNCLEAR`.
4. Cumulative prompting can overload the local model and reduce parse stability.
5. The most reliable current architecture is modular:

```text
evidence extraction
→ structured parsing
→ belief update
→ trajectory evaluation
```

---

## Recommended Next Steps

Planned next steps:

```text
1. Freeze EXAONE step-wise v2.1 as the current best local method.
2. Package the experiment as a reproducible GitHub benchmark.
3. Add HuggingFace notebook support.
4. Add optional integration tests against the upstream temporal-belief-graph package.
5. Expand scenarios with harder reversal and noisy convergence cases.
```

---

## License

Use a defensive open-source license depending on release intent.

Recommended options:

```text
Apache-2.0 for research/tooling openness
AGPL-3.0 for stronger defensive sharing requirements
```

---

## Author

Created and maintained by **CHML-real**.

GitHub:

```text
https://github.com/CHML-real
```
