# TBG-CoT-Bench Local Experiment Report

Generated at: `2026-06-23T00:48:47`

## Overview

This report summarizes a local application benchmark for temporal belief tracking.

The benchmark tests whether a system can track evidence about the temporal claim:

> Event A occurred before Event B.

The current project should be interpreted as an **application-level benchmark / usage test**, not as a full internal unit test suite for the upstream `temporal-belief-graph` package.

## Compared Methods

| Method | Parse success | Primary metric | Primary value | Secondary metrics | Coverage |
|---|---:|---|---:|---|---:|
| Rule-based baseline converter | 1.0000 | direction_accuracy | 0.5769 | strength_mae=0.1741, source_weight_mae=0.1942 | 52/52 |
| EXAONE scenario-level CoT | 0.2500 | direction_accuracy | 0.4615 | confidence_mae=0.2115, verdict_accuracy=0.6667 | 13/52 |
| EXAONE step-wise evidence v2.1 | 0.9615 | direction_accuracy | 0.6600 | confidence_mae=0.2223, verdict_accuracy=0.5000 | 50/52 |
| EXAONE order-classification v3 | 0.6154 | direction_accuracy | 1.0000 | confidence_mae=0.2960, verdict_accuracy=0.3000 | 2/52 |
| EXAONE cumulative belief v4 | 0.4423 | trajectory_verdict_accuracy | 0.3654 | p_forward_mae=0.2514 | 23/52 |

## Main Finding

The strongest current method is:

**EXAONE step-wise evidence v2.1**

It achieved:

- parse success rate: `0.9615`
- direction accuracy: `0.6600`
- rule-based baseline direction accuracy: `0.5769`

This suggests that local EXAONE becomes useful when the task is decomposed into small structured evidence judgments.

## Negative Result

The cumulative v4 method was conceptually closer to belief trajectory tracking, but performed worse in this run:

- cumulative v4 parse success rate: `0.4423`
- cumulative v4 trajectory verdict accuracy: `0.3654`

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

## Step-wise v2.1 Scenario Summary

| Scenario | Final p_forward | Verdict | Parse / coverage |
|---|---:|---|---|
| SC01 | 0.9057 | forward | 5/5, rate=1.0 |
| SC02 | 0.8876 | forward | 5/5, rate=1.0 |
| SC03 | 0.8689 | forward | 5/5, rate=1.0 |
| SC04 | 0.8312 | forward | 4/5, rate=0.8 |
| SC05 | 0.8622 | forward | 6/6, rate=1.0 |
| SC06 | 0.7912 | forward | 5/5, rate=1.0 |
| SC07 | 0.8772 | forward | 5/5, rate=1.0 |
| SC08 | 0.9089 | forward | 5/5, rate=1.0 |
| SC09 | 0.5788 | ambiguous | 4/5, rate=0.8 |
| SC10 | 0.8853 | forward | 6/6, rate=1.0 |

## Cumulative v4 Scenario Summary

| Scenario | Final p_forward | Verdict | Parse / coverage |
|---|---:|---|---|
| SC01 | 0.5000 | ambiguous |  |
| SC02 | 0.5000 | backward |  |
| SC03 | 0.5000 | ambiguous |  |
| SC04 | 0.0440 | backward |  |
| SC05 | 0.0430 | backward |  |
| SC06 | 0.5000 | ambiguous |  |
| SC07 | 0.7560 | forward |  |
| SC08 | 0.5000 | ambiguous |  |
| SC09 | 0.5000 | ambiguous |  |
| SC10 | 0.0430 | backward |  |

## Generated Assets

- `figures/converter_direction_accuracy.png` — available
- `figures/gold_trajectories.png` — available
- `figures/stepwise_vs_baseline_accuracy.png` — available
- `figures/stepwise_parse_success.png` — available
- `figures/order_v3_accuracy_comparison.png` — available
- `figures/order_v3_parse_success_comparison.png` — available
- `figures/cumulative_v4_accuracy_comparison.png` — available

## Recommended Next Step

Freeze **EXAONE step-wise evidence v2.1** as the current best local method.

Next development should focus on:

- application-level regression tests
- reproducible reports
- HuggingFace notebook packaging
- optional integration tests against the actual `temporal-belief-graph` package

