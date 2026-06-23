---
license: cc-by-nc-4.0
language:
  - en
tags:
  - temporal-reasoning
  - belief-tracking
  - chain-of-thought
  - llm-evaluation
  - ollama
  - exaone
  - benchmark
pretty_name: TBG-CoT-Bench
size_categories:
  - n<1K
task_categories:
  - text-classification
  - question-answering
---

# Dataset Card: TBG-CoT-Bench

## Dataset Summary

**TBG-CoT-Bench** contains 10 manually labeled temporal-order reasoning scenarios. Each scenario defines two events and a sequence of reasoning/evidence steps. The goal is to track the belief trajectory for whether `event_a` happened before `event_b`.

The benchmark is designed for controlled experiments on **temporal belief tracking over Chain-of-Thought-style evidence**. It includes scenario JSON files, gold evidence labels, rule-based baseline outputs, local EXAONE/Ollama outputs, trajectory CSV files, summary evaluation tables, and static visualizations.

---

## Task

Given a sequence of reasoning or evidence steps, a system should:

1. infer whether each step supports `event_a before event_b`, supports the reverse order, or is unclear;
2. estimate the strength of the evidence;
3. account for source reliability;
4. update the probability `p_forward` step by step;
5. produce a belief trajectory over the temporal claim.

The main target claim is:

```text
Event A occurred before Event B.
```

---

## Dataset Structure

```text
tbg-cot-bench/
├── scenarios/              # 10 temporal-order reasoning scenarios
├── results/                # CSV / JSONL evaluation and trajectory outputs
├── figures/                # Static result visualizations
├── notebooks/              # Notebook for inspecting and visualizing results
├── reports/                # Generated markdown experiment report
├── scripts/                # Evaluation, parsing, plotting, and run scripts
├── tests/                  # Application-level reproducibility checks
├── README.md
└── DATASET_CARD.md
```

---

## Scenario Files

Each scenario file contains:

| Field | Description |
|---|---|
| `scenario_id` | Unique scenario identifier. |
| `event_a` | First event in the temporal claim. |
| `event_b` | Second event in the temporal claim. |
| `steps` | Sequence of reasoning/evidence items. |
| `gold` fields | Manually authored labels used for evaluation and belief updates. |

---

## Step-Level Gold Fields

Each reasoning step contains:

| Field | Description |
|---|---|
| `text` | Reasoning/evidence sentence. |
| `supports_forward` | Gold temporal direction label. |
| `strength` | Gold evidence strength. |
| `source` | Gold evidence type. |
| `source_weight` | Gold reliability weight. |

---

## Trajectory Formula

Belief updates are generated using a logit-space update:

```text
logit(p_next) = logit(p_current) + direction * strength * source_weight * learning_rate
```

Default `learning_rate`:

```text
0.4
```

Where:

| Variable | Description |
|---|---|
| `p_current` | Current probability that `event_a` occurred before `event_b`. |
| `direction` | Evidence direction: forward, reverse, or neutral. |
| `strength` | Evidence strength assigned to the step. |
| `source_weight` | Reliability weight of the evidence source. |
| `p_next` | Updated belief after applying the evidence step. |

---

## Included Result Artifacts

The dataset includes the following major result categories:

| Artifact | Description |
|---|---|
| `converter_eval_summary.csv` | Rule-based baseline converter summary. |
| `stepwise_ollama_eval_summary.csv` | Step-wise EXAONE/Ollama evaluation summary. |
| `stepwise_ollama_scenario_summary.csv` | Scenario-level step-wise method summary. |
| `order_v3_eval_summary.csv` | Order-classification prompt evaluation summary. |
| `cumulative_v4_eval_summary.csv` | Cumulative belief-prediction evaluation summary. |
| `trajectories_gold.csv` | Gold belief trajectories. |
| `trajectories_auto.csv` | Rule-based baseline trajectories. |
| `trajectories_stepwise_ollama.csv` | Step-wise EXAONE/Ollama trajectories. |
| `ollama_stepwise_evidence_raw.jsonl` | Raw model outputs for step-wise evidence extraction. |

---

## Baselines and Compared Methods

This benchmark currently compares:

1. **Rule-based baseline converter**  
   A deterministic converter using hand-written extraction rules.

2. **EXAONE scenario-level CoT**  
   Full-scenario prompting where the model returns all step judgments at once.

3. **EXAONE step-wise evidence v2.1**  
   Step-level structured extraction where the model judges one evidence item at a time.

4. **EXAONE order-classification v3**  
   Classification between `A_BEFORE_B`, `B_BEFORE_A`, and `UNCLEAR`.

5. **EXAONE cumulative belief v4**  
   Cumulative prompting where the model directly estimates the current temporal conclusion.

---

## Current Local Result Summary

| Method | Parse success rate | Direction accuracy | Notes |
|---|---:|---:|---|
| Rule-based baseline converter | N/A | 0.5769 | Deterministic baseline. |
| EXAONE scenario-level CoT | Low / unstable | 0.4615 on parsed steps | Frequent JSON parse failures. |
| EXAONE step-wise evidence v2.1 | 0.9615 | 0.6600 | Best current local method. |
| EXAONE order-classification v3 | 0.6154 | Not directly comparable | Collapsed heavily into `UNCLEAR`. |
| EXAONE cumulative belief v4 | 0.4423 | Weak trajectory agreement | Less stable in the local setup. |

The current best local method is **EXAONE step-wise evidence v2.1**.

---

## Intended Use

This dataset is intended for:

- benchmarking temporal belief tracking;
- evaluating CoT-to-evidence parsers;
- testing structured local LLM extraction pipelines;
- demonstrating temporal-belief-graph workflows;
- building notebooks for belief trajectory visualization;
- comparing rule-based and local LLM-based temporal evidence converters.

---

## Out-of-Scope Use

This dataset is **not** intended for:

- factual claims about real-world events;
- legal, medical, financial, or safety-critical temporal reasoning;
- training production models without additional validation;
- evaluating general historical truthfulness.

---

## Limitations

The scenarios are synthetic and manually authored. They are intended for controlled benchmark behavior, not factual claims about real historical, legal, medical, or ecological events.

The benchmark is small: it contains 10 scenarios and 52 step-level evidence items in the current version. Results should be interpreted as an application-level local benchmark, not as a broad general-purpose temporal-reasoning leaderboard.

Local model behavior may vary depending on the Ollama model build, quantization, sampling parameters, GPU/CPU configuration, and prompt formatting.

---

## Recommended Split

This version is small enough to use as a single evaluation set.

Future versions may add:

```text
train/dev/test split
hard reversal scenarios
higher-noise convergence cases
longer cumulative evidence chains
multi-event temporal graphs
```

---

## Reproducibility

The repository includes scripts and tests for reproducing the local benchmark workflow.

Expected application test result:

```text
13 passed
```

The test suite validates:

```text
- scenario file structure
- required result files
- step-wise v2.1 performance against the baseline
- trajectory probability bounds
- report generation
```

---

## License

This dataset and generated benchmark artifacts are released under **CC-BY-NC-4.0** unless otherwise specified.

Code-side licensing can be handled separately depending on the release target.

---

## Author

Created and maintained by **CHML-real**.

GitHub:

```text
https://github.com/CHML-real
```
