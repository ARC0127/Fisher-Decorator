#!/usr/bin/env python3
import argparse, csv, json, math
from pathlib import Path
from collections import defaultdict

SUCCESS_KEYS = [
    'evaluation/success',
    'evaluation/success_rate',
    'evaluation/goal_achieved',
    'evaluation/is_success',
]
SCORE_KEYS = [
    'evaluation/score',
    'evaluation/normalized_score',
]
ALL_KEYS = SUCCESS_KEYS + SCORE_KEYS

def load_manifest(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return rows

def load_eval(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

def find_metric_col(rows):
    hdr = set(rows[0].keys())
    for k in ALL_KEYS:
        if k in hdr:
            return k
    cands = [k for k in rows[0].keys() if k.startswith('evaluation/')]
    raise RuntimeError(f"no known evaluation metric column; candidates={cands}")

def read_metric_at_steps(eval_csv, steps_csv):
    rows = load_eval(eval_csv)
    metric = find_metric_col(rows)
    by_step = {}
    for r in rows:
        step = int(float(r['step']))
        try:
            val = float(r[metric])
        except:
            continue
        if not math.isnan(val):
            by_step[step] = val
    steps = [int(x) for x in steps_csv.split(',') if x]
    vals = []
    for s in steps:
        if s in by_step:
            vals.append(by_step[s])
    if len(vals) != len(steps):
        # fallback: use last n points <= max(steps)
        eligible = sorted((s, v) for s, v in by_step.items() if s <= max(steps))
        if len(eligible) >= len(steps):
            vals = [v for _, v in eligible[-len(steps):]]
        else:
            raise RuntimeError(f"missing expected steps in {eval_csv}; expected={steps}, found={sorted(by_step.keys())[:5]}...{sorted(by_step.keys())[-5:]}")
    return sum(vals) / len(vals), metric

def mean_std(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    return mu, var ** 0.5

def alias(env):
    rep = env
    rep = rep.replace('-singletask-v0','').replace('-v0','')
    rep = rep.replace('-v1','').replace('-v2','')
    return rep

def tex_escape(s):
    return s.replace('_', '\\_')

def summarize_offline(rows, out_dir):
    methods = []
    envs = []
    data = defaultdict(list)
    metric_name = None
    scope = rows[0]['scope']

    for r in rows:
        score, metric = read_metric_at_steps(r['eval_csv'], r['paper_eval_steps'])
        metric_name = metric_name or metric
        data[(r['env_name'], r['method'])].append(score)
        if r['method'] not in methods:
            methods.append(r['method'])
        if r['env_name'] not in envs:
            envs.append(r['env_name'])

    methods = sorted(methods, key=lambda x: (0 if x == 'deflow' else 1, x))
    csv_path = out_dir / f'{scope}_offline_summary.csv'
    md_path = out_dir / f'{scope}_offline_summary.md'
    tex_path = out_dir / f'{scope}_offline_summary.tex'
    json_path = out_dir / f'{scope}_offline_summary.json'

    serial = {'scope': scope, 'mode': 'offline', 'metric': metric_name, 'methods': methods, 'rows': []}

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['env'] + methods)
        for env in envs:
            row = [alias(env)]
            row_json = {'env': env, 'env_alias': alias(env)}
            for m in methods:
                mu, sd = mean_std(data[(env, m)])
                row.append(f'{mu:.4f}±{sd:.4f}')
                row_json[m] = {'mean': mu, 'std': sd}
            w.writerow(row)
            serial['rows'].append(row_json)

    md = [f'# {scope} offline summary', '', f'- Metric: `{metric_name}`', '']
    md.append('| Env | ' + ' | '.join(methods) + ' |')
    md.append('|' + '---|' * (len(methods)+1))
    avg_row = []
    for env in envs:
        vals = []
        line = [alias(env)]
        for m in methods:
            mu, sd = mean_std(data[(env, m)])
            line.append(f'{mu:.2f}±{sd:.2f}')
        md.append('| ' + ' | '.join(line) + ' |')
    avg_line = ['**Avg**']
    for m in methods:
        mus = [mean_std(data[(env, m)])[0] for env in envs]
        mu, sd = mean_std(mus)
        avg_line.append(f'**{mu:.2f}±{sd:.2f}**')
    md.append('| ' + ' | '.join(avg_line) + ' |')
    md_path.write_text('\n'.join(md))

    tex = []
    tex.append(f'% {scope} offline summary')
    tex.append('\\begin{tabular}{l' + 'c'*len(methods) + '}')
    tex.append('\\toprule')
    tex.append('Env & ' + ' & '.join(tex_escape(m) for m in methods) + ' \\\\')
    tex.append('\\midrule')
    for env in envs:
        cells = [tex_escape(alias(env))]
        for m in methods:
            mu, sd = mean_std(data[(env, m)])
            cells.append(f'{mu:.2f}$\\\\pm${sd:.2f}')
        tex.append(' & '.join(cells) + ' \\\\')
    tex.append('\\midrule')
    cells = ['Avg']
    for m in methods:
        mus = [mean_std(data[(env, m)])[0] for env in envs]
        mu, sd = mean_std(mus)
        cells.append(f'{mu:.2f}$\\\\pm${sd:.2f}')
    tex.append(' & '.join(cells) + ' \\\\')
    tex.append('\\bottomrule')
    tex.append('\\end{tabular}')
    tex_path.write_text('\n'.join(tex))
    json_path.write_text(json.dumps(serial, indent=2))
    print(f'[OK] wrote {csv_path}')
    print(f'[OK] wrote {md_path}')
    print(f'[OK] wrote {tex_path}')
    print(f'[OK] wrote {json_path}')

def summarize_online(rows, out_dir):
    methods = []
    envs = []
    data = defaultdict(list)
    metric_name = None
    scope = rows[0]['scope']

    for r in rows:
        start_score, m1 = read_metric_at_steps(r['start_eval_csv'], r['start_steps'])
        end_score, m2 = read_metric_at_steps(r['eval_csv'], r['end_steps'])
        metric_name = metric_name or m1
        if m1 != m2:
            raise RuntimeError(f'metric mismatch for {r["env_name"]}: {m1} vs {m2}')
        data[(r['env_name'], r['method'])].append((start_score, end_score))
        if r['method'] not in methods:
            methods.append(r['method'])
        if r['env_name'] not in envs:
            envs.append(r['env_name'])

    methods = sorted(methods, key=lambda x: (0 if x == 'deflow' else 1, x))
    csv_path = out_dir / f'{scope}_online_summary.csv'
    md_path = out_dir / f'{scope}_online_summary.md'
    tex_path = out_dir / f'{scope}_online_summary.tex'
    json_path = out_dir / f'{scope}_online_summary.json'

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['env'] + methods)
        for env in envs:
            row = [alias(env)]
            for m in methods:
                starts = [x for x, _ in data[(env, m)]]
                ends = [y for _, y in data[(env, m)]]
                s_mu, s_sd = mean_std(starts)
                e_mu, e_sd = mean_std(ends)
                row.append(f'{s_mu:.4f}±{s_sd:.4f}->{e_mu:.4f}±{e_sd:.4f}')
            w.writerow(row)

    md = [f'# {scope} online summary', '', f'- Metric: `{metric_name}`', '- Protocol: report 1M -> 2M.', '']
    md.append('| Env | ' + ' | '.join(methods) + ' |')
    md.append('|' + '---|' * (len(methods)+1))
    for env in envs:
        line = [alias(env)]
        for m in methods:
            starts = [x for x, _ in data[(env, m)]]
            ends = [y for _, y in data[(env, m)]]
            s_mu, s_sd = mean_std(starts)
            e_mu, e_sd = mean_std(ends)
            line.append(f'{s_mu:.2f}±{s_sd:.2f}→{e_mu:.2f}±{e_sd:.2f}')
        md.append('| ' + ' | '.join(line) + ' |')
    md_path.write_text('\n'.join(md))

    tex = []
    tex.append(f'% {scope} online summary')
    tex.append('\\begin{tabular}{l' + 'c'*len(methods) + '}')
    tex.append('\\toprule')
    tex.append('Env & ' + ' & '.join(tex_escape(m) for m in methods) + ' \\\\')
    tex.append('\\midrule')
    for env in envs:
        cells = [tex_escape(alias(env))]
        for m in methods:
            starts = [x for x, _ in data[(env, m)]]
            ends = [y for _, y in data[(env, m)]]
            s_mu, s_sd = mean_std(starts)
            e_mu, e_sd = mean_std(ends)
            cells.append(f'{s_mu:.2f}$\\\\pm${s_sd:.2f}$\\\\to${e_mu:.2f}$\\\\pm${e_sd:.2f}')
        tex.append(' & '.join(cells) + ' \\\\')
    tex.append('\\bottomrule')
    tex.append('\\end{tabular}')
    tex_path.write_text('\n'.join(tex))
    json_path.write_text(json.dumps({'scope': scope, 'mode': 'online', 'metric': metric_name}, indent=2))
    print(f'[OK] wrote {csv_path}')
    print(f'[OK] wrote {md_path}')
    print(f'[OK] wrote {tex_path}')
    print(f'[OK] wrote {json_path}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--mode', choices=['offline', 'online'], required=True)
    args = ap.parse_args()
    rows = load_manifest(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == 'offline':
        summarize_offline(rows, out_dir)
    else:
        summarize_online(rows, out_dir)

if __name__ == '__main__':
    main()
