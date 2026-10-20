#!/usr/bin/env python3
"""Run all prepared datasets with matched windows and optional LIFT ablation.

Examples (from the repository root):
  python -u scripts/run_all.py --device cuda:0
  python -u scripts/run_all.py --device cuda:0 --no-lift
  python -u scripts/run_all.py --device cuda:0 --variants y-base patch-base patch-lift

Default: dataset-short presets (BE/NP 168h->24h; DIST 2h->30min).
The uniform profile explicitly retains the old 24->24 native-step protocol.
Window alignment does not reproduce DAG's split/rolling/loss protocol.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The 12 Table-3 datasets plus DIST. Steps are physical seconds, not row counts.
STEPS = {'DIST': 3600, 'NP': 3600, 'PJM': 3600, 'BE': 3600, 'FR': 3600,
         'DE': 3600, 'Energy': 3600, 'Sdwpfm1': 1800, 'Sdwpfm2': 1800,
         'Sdwpfh1': 3600, 'Sdwpfh2': 3600, 'Colbun': 86400, 'Rapel': 86400}
VARIANTS = ('y-base', 'patch-base', 'patch-lift', 'dag-lift', 'dag-history', 'dag')
SIMPLE = ('y-base', 'patch-base', 'patch-lift')
OVERRIDES = ('epochs', 'patience', 'batch_size', 'd_model', 'n_heads', 'n_layers',
             'state_num', 'lr', 'alpha', 'beta')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT, help='Repository root')
    p.add_argument('--suite', choices=['all', 'table3'], default='all',
                   help='all = DIST + 12 Table-3 datasets; table3 excludes DIST')
    p.add_argument('--datasets', nargs='+', choices=list(STEPS), help='Run only these datasets')
    p.add_argument('--variants', nargs='+', choices=VARIANTS,
                   help='Default patch-lift; y-base/patch-base are component baselines; DAG variants retained')
    p.add_argument('--no-lift', action='store_true', help='Default patch-base; explicit dag-lift -> dag-history')
    p.add_argument('--device', default='cuda' if _cuda_available() else 'cpu')
    p.add_argument('--data-dir', type=Path, help='Optional flat directory containing DATASET.npz')
    p.add_argument('--output-dir', type=Path, help='New output directory; default is unique')
    p.add_argument('--protocol', choices=['dataset-short', 'dataset-long', 'uniform'],
                   default='dataset-short', help='Use dataset-specific windows; uniform explicitly retains old 24->24')
    p.add_argument('--history-steps', type=int, help='uniform only; default 24 native steps')
    p.add_argument('--forecast-steps', type=int, help='uniform only; default 24 native steps')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--patch-encoder', choices=['ttcn', 'mean'], default='ttcn')
    p.add_argument('--lag-estimator', choices=['fft', 'pearson'], default='fft')
    p.add_argument('--refiner-mode', choices=['global', 'prefix'], default='global')
    for name in OVERRIDES:
        p.add_argument('--'+name.replace('_', '-'), type=float if name in ('lr', 'alpha', 'beta') else int)
    p.add_argument('--dry-run', action='store_true', help='Validate every dataset and print commands; no outputs')
    p.add_argument('--smoke', action='store_true', help='Eight records, one epoch, small model; not accuracy results')
    p.add_argument('--fail-fast', action='store_true', help='Stop after the first failed training job')
    return p


def _cuda_available():
    import torch
    return torch.cuda.is_available()


def resolve(root, path):
    return path.resolve() if path.is_absolute() else (root/path).resolve()


def selections(args):
    datasets = args.datasets or [name for name in STEPS if args.suite == 'all' or name != 'DIST']
    variants = args.variants or ['patch-lift']
    if args.no_lift:
        without = {'patch-lift': 'patch-base', 'patch-base': 'patch-base',
                   'dag-lift': 'dag-history', 'dag-history': 'dag-history'}
        if len(variants) != 1 or variants[0] not in without:
            raise ValueError('--no-lift cannot be combined with other --variants')
        variants = [without[variants[0]]]
    if len(datasets) != len(set(datasets)) or len(variants) != len(set(variants)):
        raise ValueError('Duplicate datasets/variants would overwrite a job; remove duplicates')
    return datasets, variants


def task_settings(root, args, name):
    """Resolve physical durations before inspecting data or constructing jobs."""
    if args.protocol != 'uniform' and (args.history_steps is not None or args.forecast_steps is not None):
        raise ValueError('Step overrides require --protocol uniform; dataset protocols read their JSON windows')
    if args.protocol == 'dataset-long':
        if name == 'DIST':
            raise ValueError('DIST is not in DAG; no paper long-term preset. Use dataset-short or explicitly use uniform')
        config_path = root/'configs/table3_irregular'/f'{name}_long.json'
    else:
        folder = 'configs' if name in ('DIST', 'BE', 'NP') else 'configs/table3_irregular'
        config_path = root/folder/f'{name}.json'
    config = json.loads(config_path.read_text())
    if args.protocol == 'uniform':
        patch = STEPS[name]
        history = (args.history_steps if args.history_steps is not None else 24)*patch
        forecast = (args.forecast_steps if args.forecast_steps is not None else 24)*patch
    else:
        history, forecast, patch = (config[k] for k in ('history', 'forecast', 'patch_seconds'))
    if (min(history, forecast, patch) <= 0 or history % patch or forecast % patch):
        raise ValueError('Preset durations must be positive whole patch multiples')
    return config_path, config, {
        'protocol': args.protocol, 'step_seconds': patch,
        'history_steps': int(history//patch), 'forecast_steps': int(forecast//patch),
        'history_seconds': history, 'forecast_seconds': forecast,
        'task_id': f'{name}_h{history:g}_f{forecast:g}_p{patch:g}',
        'task_reference': ('explicit uniform window; not DAG paper protocol' if args.protocol == 'uniform'
            else 'custom DIST initial industrial task; not in DAG paper' if name == 'DIST'
            else 'DAG v3 short/long window lengths only; irregular evaluation retained')}


def preflight(root, args, datasets):
    import numpy as np
    from exobase.data import load_long_table, TimeRecords, split_records, normalization_from_seen
    from exobase.metrics import fit_evaluation_statistics
    from exobase.prepare import sha256
    plans, errors = [], []
    for name in datasets:
        try:
            config_path, config, task = task_settings(root, args, name)
            source = (resolve(root, args.data_dir)/f'{name}.npz' if args.data_dir else
                      resolve(root, Path(config['data_path'])))
            if not args.data_dir and not source.exists() and name in ('BE', 'NP'):
                source = root/'data/processed/table3_irregular'/f'{name}.npz'
            table = load_long_table(source, config['targets'], config['exog'])
            gaps = np.diff(table.time)
            native_step = STEPS[name]
            if not len(gaps) or not np.isfinite(table.time).all() or (gaps <= 0).any():
                raise ValueError('timestamps must be finite and strictly increasing')
            if name != 'DIST':
                unique, counts = np.unique(gaps, return_counts=True)
                if (not np.isclose(unique[counts.argmax()], native_step) or
                        not np.allclose(gaps/native_step, np.rint(gaps/native_step))):
                    raise ValueError(f'expected native interval {native_step}s; check timestamp units')
            history, forecast = task['history_seconds'], task['forecast_seconds']
            records = TimeRecords(table, history, forecast, len(config['targets']))
            if args.smoke:
                records.records = records.records[:8]
            split = split_records(records, 'chronological')
            if not all(split.values()):
                raise ValueError('train/validation/test split must each contain records')
            normalization_from_seen(records, split['train'])
            fit_evaluation_statistics(records, split['train'])
            record_ids = {key: [records.records[i][0] for i in ids] for key, ids in split.items()}
            plan = {'dataset': name, **task, 'valid_records': len(records),
                    'record_ids': record_ids, 'split_records': {k: len(v) for k, v in split.items()},
                    'config': str(config_path), 'data_path': str(source), 'data_sha256': sha256(source)}
            plans.append(plan)
            print(f'Checked {name}: history={history}s forecast={forecast}s '
                  f'({task["history_steps"]}->{task["forecast_steps"]} patches), '
                  f'patch={task["step_seconds"]}s, protocol={args.protocol}, '
                  f'records={len(records)}, split={plan["split_records"]}', flush=True)
        except (ValueError, OSError, KeyError) as error:
            errors.append(f'{name}: {error}')
    if errors:
        raise ValueError('Preflight failed before starting any training:\n'+'\n'.join(errors))
    return plans


def build_jobs(root, args, plans, variants, output):
    jobs = []
    for plan in plans:
        for variant in variants:
            command = [sys.executable, '-u', '-m', 'exobase.train',
                '--config', plan['config'], '--data-path', plan['data_path'],
                '--variant', variant, '--patch-encoder', 'mean' if variant == 'y-base' else args.patch_encoder,
                '--task-protocol', args.protocol,
                '--lag-estimator', args.lag_estimator, '--refiner-mode', args.refiner_mode,
                '--split', 'chronological', '--normalization', 'train',
                '--history', str(plan['history_seconds']), '--forecast', str(plan['forecast_seconds']),
                '--patch-seconds', str(plan['step_seconds']), '--patch-stride', str(plan['step_seconds']),
                '--device', args.device, '--threads', str(args.threads), '--seed', str(args.seed),
                '--save-path', str(output/plan['dataset']/variant)]
            values = {key: getattr(args, key) for key in OVERRIDES}
            if args.smoke:
                values.update(epochs=1, d_model=args.d_model or 8, n_heads=args.n_heads or 2,
                              state_num=args.state_num or 2, batch_size=args.batch_size or 2)
                command += ['--max-records', '8']
            for key, value in values.items():
                if value is not None:
                    command += ['--'+key.replace('_', '-'), str(value)]
            jobs.append({'dataset': plan['dataset'], 'variant': variant, 'command': command})
    return jobs


def run_job(command, root, log_path):
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def result_row(plan, job, args, output):
    folder = output/plan['dataset']/job['variant']
    manifest = json.loads((folder/'manifest.json').read_text())
    metrics = json.loads((folder/'metrics.json').read_text())
    settings = manifest['args']
    expected = {'variant': job['variant'], 'history': plan['history_seconds'],
                'forecast': plan['forecast_seconds'], 'patch_seconds': plan['step_seconds'],
                'patch_stride': plan['step_seconds'],
                'patch_encoder': 'mean' if job['variant'] == 'y-base' else args.patch_encoder,
                'seed': args.seed, 'split': 'chronological', 'normalization': 'train',
                'task_protocol': args.protocol,
                'lag_estimator': args.lag_estimator, 'refiner_mode': args.refiner_mode}
    for key in OVERRIDES:
        if getattr(args, key) is not None:
            expected[key] = 0. if key == 'beta' and job['variant'] in SIMPLE else getattr(args, key)
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError('Saved configuration differs from the planned comparison')
    if manifest['data_sha256'] != plan['data_sha256'] or manifest['record_ids'] != plan['record_ids']:
        raise ValueError('Dataset or record splits differ from the preflight plan')
    test = metrics['test']
    if test.get('metric_scale') != 'train_standard' or test.get('metric_reduction') != 'variable_macro':
        raise ValueError('Expected train-standard, variable-macro evaluation')
    baseline = test['baselines']['persistence']
    return {'mse': test['mse'], 'mae': test['mae'], 'rmse': test['rmse'],
            'persistence_mse': baseline['mse'], 'persistence_mae': baseline['mae'],
            'best_epoch': metrics['best_epoch'], 'metric_scale': test['metric_scale'],
            'metric_reduction': test['metric_reduction']}


def write_comparison(output, rows):
    fields = ['dataset', 'with_lift_variant', 'without_lift_variant',
              'with_lift_mse', 'without_lift_mse', 'mse_delta_with_minus_without',
              'with_lift_mae', 'without_lift_mae', 'mae_delta_with_minus_without']
    with (output/'ablation_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in dict.fromkeys(row['dataset'] for row in rows):
            pair = {row['variant']: row for row in rows if row['dataset'] == name and row['status'] == 'ok'}
            for with_lift, without_lift in [('patch-lift', 'patch-base'), ('dag-lift', 'dag-history')]:
                if not {with_lift, without_lift} <= pair.keys():
                    continue
                full, base = pair[with_lift], pair[without_lift]
                writer.writerow({'dataset': name, 'with_lift_variant': with_lift,
                    'without_lift_variant': without_lift,
                    'with_lift_mse': full['mse'], 'without_lift_mse': base['mse'],
                    'mse_delta_with_minus_without': full['mse']-base['mse'],
                    'with_lift_mae': full['mae'], 'without_lift_mae': base['mae'],
                    'mae_delta_with_minus_without': full['mae']-base['mae']})
    fields = ['dataset', 'reference_variant', 'variant', 'mse_delta', 'mae_delta']
    with (output/'component_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in dict.fromkeys(row['dataset'] for row in rows):
            pair = {row['variant']: row for row in rows if row['dataset'] == name and row['status'] == 'ok'}
            for reference, variant in [('y-base', 'patch-base'), ('patch-base', 'patch-lift'),
                                       ('patch-base', 'dag'), ('patch-lift', 'dag')]:
                if {reference, variant} <= pair.keys():
                    writer.writerow({'dataset': name, 'reference_variant': reference, 'variant': variant,
                        'mse_delta': pair[variant]['mse']-pair[reference]['mse'],
                        'mae_delta': pair[variant]['mae']-pair[reference]['mae']})


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if any(v is not None and v < 1 for v in (args.history_steps, args.forecast_steps, args.threads)):
        p.error('history-steps, forecast-steps and threads must be positive')
    if args.smoke and args.epochs not in (None, 1):
        p.error('--smoke runs exactly one epoch')
    for key in OVERRIDES:
        value = getattr(args, key)
        if value is not None and ((key in ('alpha', 'beta') and value < 0) or
                                 (key not in ('alpha', 'beta') and value <= 0)):
            p.error(f'Invalid --{key.replace("_", "-")}')
    if args.alpha is not None and args.alpha > 1:
        p.error('alpha must be in [0,1]')
    root = args.root.resolve()
    try:
        datasets, variants = selections(args)
        if {'patch-base', 'patch-lift'} & set(variants) and args.patch_encoder != 'ttcn':
            raise ValueError('patch-base/patch-lift require --patch-encoder ttcn')
        plans = preflight(root, args, datasets)
        if ({'dag-lift', 'patch-lift'} & set(variants) and args.lag_estimator == 'fft'
                and any(plan['history_steps'] < 3 for plan in plans)):
            raise ValueError('FFT local peaks need at least 3 history patch bins')
    except (ValueError, OSError, KeyError) as error:
        p.error(str(error))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    group = (f'uniform_{args.history_steps or 24}to{args.forecast_steps or 24}'
             if args.protocol == 'uniform' else args.protocol)
    output = (resolve(root, args.output_dir) if args.output_dir else
              root/'results'/group/
              f'{"SMOKE" if args.smoke else "run"}_{stamp}')
    if output.exists() and not args.dry_run:
        p.error(f'Output already exists: {output}; choose a new directory')
    jobs = build_jobs(root, args, plans, variants, output)
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job['command']), flush=True)
        print(f'Dry run complete: {len(jobs)} jobs; no training or output files created.', flush=True)
        return 0
    output.mkdir(parents=True)
    (output/'plan.json').write_text(json.dumps({'smoke_test': args.smoke, 'seed': args.seed,
        'protocol': args.protocol, 'window_alignment_only': args.protocol != 'uniform',
        'datasets': plans, 'variants': variants, 'jobs': jobs}, ensure_ascii=False, indent=2))
    print(f'Running {len(jobs)} sequential jobs. Results: {output}', flush=True)
    fields = ['dataset', 'variant', 'status', 'seed', 'protocol', 'history_steps', 'forecast_steps',
              'history_seconds', 'forecast_seconds', 'step_seconds',
              'mse', 'mae', 'rmse', 'persistence_mse', 'persistence_mae',
              'best_epoch', 'metric_scale', 'metric_reduction', 'smoke_test', 'log', 'metrics']
    rows = []
    by_name = {plan['dataset']: plan for plan in plans}
    with (output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        for job in jobs:
            plan = by_name[job['dataset']]
            log_path = output/f'{job["dataset"]}_{job["variant"]}.log'
            row = {key: plan[key] for key in ('dataset', 'protocol', 'history_steps', 'forecast_steps',
                'history_seconds', 'forecast_seconds', 'step_seconds')}
            row.update(variant=job['variant'], seed=args.seed, smoke_test=args.smoke,
                       log=str(log_path), metrics=str(output/job['dataset']/job['variant']/'metrics.json'))
            print('\n'+shlex.join(job['command']), flush=True)
            try:
                code = run_job(job['command'], root, log_path)
                if code:
                    raise ValueError(f'training exited with code {code}')
                row.update(result_row(plan, job, args, output), status='ok')
            except (ValueError, OSError, KeyError) as error:
                row['status'] = f'failed: {error}'
                print(f'{job["dataset"]}/{job["variant"]}: {row["status"]}', flush=True)
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            write_comparison(output, rows)
            if row['status'] != 'ok' and args.fail_fast:
                break
    failed = [row for row in rows if row['status'] != 'ok']
    print(f'\nSummary: {output/"summary.csv"}; failed jobs: {len(failed)}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
