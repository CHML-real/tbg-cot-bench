# Dataset Card: TBG-CoT-Bench

## Dataset summary

TBG-CoT-Bench contains 10 manually labeled temporal-order reasoning scenarios. Each scenario defines two events and a sequence of reasoning/evidence steps. The goal is to track the belief trajectory for whether `event_a` happened before `event_b`.

## Task

Given a sequence of reasoning steps, estimate evidence labels and update the probability `p_forward` step by step.

## Gold fields

Each step contains:

- `text`: reasoning/evidence sentence
- `supports_forward`: gold temporal direction label
- `strength`: gold evidence strength
- `source`: gold evidence type
- `source_weight`: gold reliability weight

## Trajectory formula

```text
logit(p_next) = logit(p_current) + direction * strength * source_weight * learning_rate
```

Default `learning_rate`: **0.4**.

## Intended use

- Benchmarking temporal belief tracking
- Evaluating CoT-to-evidence parsers
- Demonstrating temporal-belief-graph workflows
- Creating HuggingFace notebooks for belief trajectory visualization

## Limitations

The scenarios are synthetic and manually authored. They are intended for controlled benchmark behavior, not factual claims about real historical, legal, medical, or ecological events.

## Recommended split

This version is small enough to use as a single evaluation set. Future versions may add train/dev/test splits.
