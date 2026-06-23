#!/usr/bin/env python
from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row[key], []).append(row)
    return out


def verdict_band(ax):
    ax.axhline(0.65, linestyle='--', linewidth=1)
    ax.axhline(0.35, linestyle='--', linewidth=1)
    ax.axhline(0.50, linestyle=':', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('p_forward')
    ax.set_xlabel('CoT / evidence step')


def plot_gold_trajectories(gold_rows: list[dict], out_dir: Path) -> None:
    grouped = group_by(gold_rows, 'scenario_id')
    fig, ax = plt.subplots(figsize=(11, 6))
    for sid, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: int(r['step']))
        xs = [int(r['step']) for r in rows]
        ys = [float(r['p_forward']) for r in rows]
        ax.plot(xs, ys, marker='o', linewidth=1.6, label=sid)
    verdict_band(ax)
    ax.set_title('Gold-label belief trajectories')
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'gold_trajectories.png', dpi=180)
    plt.close(fig)


def plot_final_p(summary_rows: list[dict], out_dir: Path) -> None:
    rows = sorted(summary_rows, key=lambda r: r['scenario_id'])
    labels = [r['scenario_id'] for r in rows]
    vals = [float(r['final_p']) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, vals)
    ax.axhline(0.65, linestyle='--', linewidth=1)
    ax.axhline(0.35, linestyle='--', linewidth=1)
    ax.axhline(0.50, linestyle=':', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Final p_forward')
    ax.set_title('Final belief by scenario')
    fig.tight_layout()
    fig.savefig(out_dir / 'final_p_by_scenario.png', dpi=180)
    plt.close(fig)


def plot_converter_accuracy(eval_rows: list[dict], out_dir: Path) -> None:
    grouped = group_by(eval_rows, 'scenario_id')
    labels, vals = [], []
    for sid, rows in sorted(grouped.items()):
        usable = [r for r in rows if r.get('auto_supports_forward') not in ('', 'None', None)]
        if not usable:
            acc = 0.0
        else:
            acc = sum(1 for r in usable if r.get('direction_correct') == 'True') / len(usable)
        labels.append(sid)
        vals.append(acc)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, vals)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Direction accuracy')
    ax.set_title('Baseline converter direction accuracy')
    fig.tight_layout()
    fig.savefig(out_dir / 'converter_direction_accuracy.png', dpi=180)
    plt.close(fig)


def plot_gold_vs_auto(gold_rows: list[dict], auto_rows: list[dict], out_dir: Path) -> None:
    gold = group_by(gold_rows, 'scenario_id')
    auto = group_by(auto_rows, 'scenario_id')
    scenario_dir = out_dir / 'scenario_plots'
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for sid in sorted(gold.keys()):
        g_rows = sorted(gold[sid], key=lambda r: int(r['step']))
        a_rows = sorted(auto.get(sid, []), key=lambda r: int(r['step']))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot([int(r['step']) for r in g_rows], [float(r['p_forward']) for r in g_rows], marker='o', label='gold')
        if a_rows:
            ax.plot([int(r['step']) for r in a_rows], [float(r['p_forward']) for r in a_rows], marker='x', label='auto')
        verdict_band(ax)
        ax.set_title(f'{sid}: gold vs auto trajectory')
        ax.legend()
        fig.tight_layout()
        fig.savefig(scenario_dir / f'{sid.lower()}_gold_vs_auto.png', dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize TBG-CoT-Bench result CSV files.')
    parser.add_argument('--results', default='results')
    parser.add_argument('--figures', default='figures')
    args = parser.parse_args()

    results = Path(args.results)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)

    gold_rows = read_csv(results / 'trajectories_gold.csv')
    summary_rows = read_csv(results / 'scenario_summary.csv')
    auto_rows = read_csv(results / 'trajectories_auto.csv') if (results / 'trajectories_auto.csv').exists() else []
    eval_rows = read_csv(results / 'converter_eval.csv') if (results / 'converter_eval.csv').exists() else []

    plot_gold_trajectories(gold_rows, figures)
    plot_final_p(summary_rows, figures)
    if eval_rows:
        plot_converter_accuracy(eval_rows, figures)
    if auto_rows:
        plot_gold_vs_auto(gold_rows, auto_rows, figures)

    print(f'Wrote figures to {figures}')


if __name__ == '__main__':
    main()
