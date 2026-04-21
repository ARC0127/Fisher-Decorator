#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple


def read_manifest(path: Path) -> List[Dict[str, str]]:
    rows = []
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open('r', newline='') as f:
        return list(csv.DictReader(f))


def find_metric_column(rows: List[Dict[str, str]]) -> str:
    if not rows:
        raise RuntimeError("eval.csv is empty")
    preferred = [
        'evaluation/success',
        'evaluation/success_rate',
        'evaluation/goal_achieved',
        'evaluation/score',
        'evaluation/normalized_score',
    ]
    header = set(rows[0].keys())
    for key in preferred:
        if key in header:
            return key
    candidates = [k for k in rows[0].keys() if k.startswith('evaluation/')]
    raise RuntimeError(f"Could not identify success metric. Available evaluation columns: {candidates}")


def parse_float(x: str) -> float:
    if x is None or x == '':
        return float('nan')
    return float(x)


def summarize_eval(eval_csv: Path, expected_steps: List[int]) -> Tuple[float, str, List[Tuple[int, float]]]:
    rows = load_csv_rows(eval_csv)
    metric_col = find_metric_column(rows)
    by_step = {}
    for row in rows:
        step = int(float(row['step']))
        metric = parse_float(row[metric_col])
        if not math.isnan(metric):
            by_step[step] = metric
    selected = []
    for step in expected_steps:
        if step in by_step:
            selected.append((step, by_step[step]))
    if len(selected) < 3:
        # fall back to last three eval steps <= max expected step, excluding step 1 unless necessary
        eligible = sorted((s, v) for s, v in by_step.items() if s <= max(expected_steps) and s != 1)
        if len(eligible) >= 3:
            selected = eligible[-3:]
        else:
            eligible = sorted((s, v) for s, v in by_step.items())
            if len(eligible) < 3:
                raise RuntimeError(f"Not enough eval points in {eval_csv}; found {len(eligible)}")
            selected = eligible[-3:]
    score = sum(v for _, v in selected) / len(selected)
    step_str = ','.join(str(s) for s, _ in selected)
    return score, metric_col, selected


def tex_escape(s: str) -> str:
    return s.replace('_', '\\_')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path)
    methods = []
    envs = []
    scores: Dict[Tuple[str, str], float] = {}
    steps_used: Dict[Tuple[str, str], str] = {}
    metric_name = None

    for row in rows:
        env = row['env_name']
        method = row['method']
        eval_csv = Path(row['eval_csv'])
        expected_steps = [int(x) for x in row['paper_eval_steps'].split(',') if x]
        score, cur_metric, selected = summarize_eval(eval_csv, expected_steps)
        metric_name = metric_name or cur_metric
        if cur_metric != metric_name:
            raise RuntimeError(f"Metric mismatch: {cur_metric} vs {metric_name}")
        if method not in methods:
            methods.append(method)
        if env not in envs:
            envs.append(env)
        scores[(env, method)] = score
        steps_used[(env, method)] = ','.join(str(s) for s, _ in selected)

    if len(methods) != 2:
        raise RuntimeError(f"Expected exactly 2 methods (baseline + variant), got {methods}")

    baseline, variant = methods[0], methods[1]
    env_alias = {
        'antmaze-large-navigate-singletask-task1-v0': 'antmaze-large',
        'antmaze-giant-navigate-singletask-task1-v0': 'antmaze-giant',
        'humanoidmaze-medium-navigate-singletask-task1-v0': 'humanoidmaze-medium',
        'humanoidmaze-large-navigate-singletask-task1-v0': 'humanoidmaze-large',
        'antsoccer-arena-navigate-singletask-task4-v0': 'antsoccer',
        'cube-single-play-singletask-v0': 'cube-single',
        'cube-double-play-singletask-v0': 'cube-double',
        'scene-play-singletask-v0': 'scene',
        'puzzle-3x3-play-singletask-v0': 'puzzle-3x3',
        'puzzle-4x4-play-singletask-v0': 'puzzle-4x4',
    }

    csv_path = out_dir / 'ogbench_state5_1seed_summary.csv'
    md_path = out_dir / 'ogbench_state5_1seed_summary.md'
    tex_path = out_dir / 'ogbench_state5_1seed_summary.tex'
    json_path = out_dir / 'ogbench_state5_1seed_summary.json'

    serializable = {
        'metric_name': metric_name,
        'baseline': baseline,
        'variant': variant,
        'rows': [],
    }

    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['env', baseline, variant, 'delta_variant_minus_baseline', 'baseline_steps', 'variant_steps'])
        bvals = []
        vvals = []
        for env in envs:
            b = scores[(env, baseline)]
            v = scores[(env, variant)]
            bvals.append(b)
            vvals.append(v)
            writer.writerow([
                env_alias.get(env, env),
                f'{b:.4f}',
                f'{v:.4f}',
                f'{v - b:+.4f}',
                steps_used[(env, baseline)],
                steps_used[(env, variant)],
            ])
            serializable['rows'].append({
                'env': env,
                'env_alias': env_alias.get(env, env),
                baseline: b,
                variant: v,
                'delta_variant_minus_baseline': v - b,
                'baseline_steps': steps_used[(env, baseline)],
                'variant_steps': steps_used[(env, variant)],
            })
        writer.writerow(['Avg', f'{sum(bvals)/len(bvals):.4f}', f'{sum(vvals)/len(vvals):.4f}', f'{sum(vvals)/len(vvals) - sum(bvals)/len(bvals):+.4f}', '', ''])
        serializable['avg'] = {
            baseline: sum(bvals) / len(bvals),
            variant: sum(vvals) / len(vvals),
            'delta_variant_minus_baseline': sum(vvals) / len(vvals) - sum(bvals) / len(bvals),
        }

    md_lines = []
    md_lines.append(f'# OGBench state-5 summary (1 seed)')
    md_lines.append('')
    md_lines.append(f'- Metric: `{metric_name}`')
    md_lines.append('- Paper protocol: average of the last three evaluation epochs for state-based OGBench (800k, 900k, 1M).')
    md_lines.append('')
    md_lines.append(f'| Env | {baseline} | {variant} | Δ (variant-baseline) |')
    md_lines.append('|---|---:|---:|---:|')
    bvals = []
    vvals = []
    for env in envs:
        alias = env_alias.get(env, env)
        b = scores[(env, baseline)]
        v = scores[(env, variant)]
        bvals.append(b)
        vvals.append(v)
        md_lines.append(f'| {alias} | {b:.4f} | {v:.4f} | {v - b:+.4f} |')
    md_lines.append(f'| **Avg** | **{sum(bvals)/len(bvals):.4f}** | **{sum(vvals)/len(vvals):.4f}** | **{sum(vvals)/len(vvals) - sum(bvals)/len(bvals):+.4f}** |')
    md_lines.append('')
    md_lines.append('Steps actually used for averaging are stored in the CSV/JSON sidecars.')
    md_path.write_text('\n'.join(md_lines))

    tex_lines = []
    tex_lines.append('% OGBench state-5, 1 seed, paper protocol = average of last 3 eval epochs (800k/900k/1M)')
    tex_lines.append('\\begin{tabular}{lccc}')
    tex_lines.append('\\toprule')
    tex_lines.append(f'Env & {tex_escape(baseline)} & {tex_escape(variant)} & $\\Delta$ \\\\')
    tex_lines.append('\\midrule')
    bvals = []
    vvals = []
    for env in envs:
        alias = tex_escape(env_alias.get(env, env))
        b = scores[(env, baseline)]
        v = scores[(env, variant)]
        bvals.append(b)
        vvals.append(v)
        tex_lines.append(f'{alias} & {b:.4f} & {v:.4f} & {v - b:+.4f} \\\\')
    tex_lines.append('\\midrule')
    tex_lines.append(f'Avg & {sum(bvals)/len(bvals):.4f} & {sum(vvals)/len(vvals):.4f} & {sum(vvals)/len(vvals) - sum(bvals)/len(bvals):+.4f} \\\\')
    tex_lines.append('\\bottomrule')
    tex_lines.append('\\end{tabular}')
    tex_path.write_text('\n'.join(tex_lines))

    json_path.write_text(json.dumps(serializable, indent=2))

    print(f'[OK] wrote {csv_path}')
    print(f'[OK] wrote {md_path}')
    print(f'[OK] wrote {tex_path}')
    print(f'[OK] wrote {json_path}')
    print(f'[INFO] metric={metric_name} baseline={baseline} variant={variant}')


if __name__ == '__main__':
    main()
