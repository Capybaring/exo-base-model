#!/usr/bin/env python3
"""Run the nine remaining Table-3 datasets: DAG, no LIFT, 24 native steps -> 24.

Run from the repository root: python -u scripts/run_remaining_dag.py --device cuda:1
The prepared NPZ timestamps are seconds; native steps differ across datasets.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np


# Verified against the prepared NPZ timestamps and dataset_summary.csv.
DATASETS = [
    ('FR', 3600), ('DE', 3600), ('Energy', 3600),
    ('Sdwpfm1', 1800), ('Sdwpfm2', 1800),
    ('Sdwpfh1', 3600), ('Sdwpfh2', 3600),
    ('Colbun', 86400), ('Rapel', 86400),
]


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root/path


def preflight(root, config_dir, data_dir):
    sys.path.insert(0, str(root))
    try:
        from exobase.data import load_long_table, TimeRecords, split_records
        from exobase.metrics import fit_evaluation_statistics
    except ImportError as error:
        raise ValueError('Run inside exo-base-model and install the three-file evaluation fix first.') from error
    plans = []
    for name, step in DATASETS:
        config_path = config_dir/f'{name}.json'
        config = json.loads(config_path.read_text())
        source = data_dir/f'{name}.npz' if data_dir else resolve(root, config['data_path'])
        table = load_long_table(source, config['targets'], config['exog'])
        gaps = np.diff(table.time)
        if not len(gaps) or not np.isfinite(table.time).all() or (gaps <= 0).any():
            raise ValueError(f'{name}: timestamps must be finite and strictly increasing')
        unique, counts = np.unique(gaps, return_counts=True)
        mode = float(unique[counts.argmax()])
        if (not np.isclose(mode, step) or not np.isclose(gaps.min(), step)
                or not np.allclose(gaps/step, np.rint(gaps/step))):
            raise ValueError(f'{name}: expected base interval {step} seconds, '
                             f'but minimum={gaps.min()}, most frequent={mode}; check timestamp units')
        duration = 24*step
        records = TimeRecords(table, duration, duration, len(config['targets']))
        split = split_records(records, 'chronological')
        # Fail before launching jobs if training-only evaluation scaling cannot be fit.
        fit_evaluation_statistics(records, split['train'])
        plan = {'dataset': name, 'step_seconds': step, 'history_steps': 24,
                'forecast_steps': 24, 'history_seconds': duration,
                'forecast_seconds': duration, 'valid_records': len(records),
                'split_records': {key: len(ids) for key, ids in split.items()},
                'config': str(config_path), 'data_path': str(source)}
        plans.append(plan)
        print(f'Checked {name}: timestamp unit=seconds, base interval={step}s, '
              f'24 -> 24 steps, history/forecast={duration}s each, '
              f'records={len(records)}, split={plan["split_records"]}', flush=True)
    return plans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Repository root')
    parser.add_argument('--config-dir', default='configs/table3_irregular')
    parser.add_argument('--data-dir', help='Optional directory containing the nine prepared NPZ files')
    parser.add_argument('--output-dir', help='New output directory; default includes a unique run timestamp')
    parser.add_argument('--epochs', type=int, help='Override epochs; otherwise retain training defaults/presets')
    parser.add_argument('--batch-size', type=int, help='Override batch size; otherwise retain each preset')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true', help='Validate all data and print commands only')
    parser.add_argument('--smoke', action='store_true', help='One epoch, eight records, small model; NOT final results')
    args = parser.parse_args(argv)
    if args.threads < 1 or (args.epochs is not None and args.epochs < 1) or (
            args.batch_size is not None and args.batch_size < 1):
        parser.error('threads, epochs and batch-size must be positive')
    if args.smoke and args.epochs not in (None, 1):
        parser.error('--smoke runs exactly one epoch')
    root = args.root.resolve()
    config_dir = resolve(root, args.config_dir)
    data_dir = resolve(root, args.data_dir) if args.data_dir else None
    try:
        plans = preflight(root, config_dir, data_dir)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    if args.output_dir:
        output = resolve(root, args.output_dir)
    else:
        label = 'SMOKE' if args.smoke else 'run'
        output = root/'results'/'remaining_dag_24to24'/f'{label}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if output.exists() and not args.dry_run:
        parser.error(f'Output directory already exists: {output}; choose a new directory')
    commands = []
    for plan in plans:
        name = plan['dataset']
        command = [sys.executable, '-u', '-m', 'exobase.train',
            '--config', plan['config'], '--data-path', plan['data_path'],
            '--variant', 'dag', '--split', 'chronological', '--normalization', 'train',
            '--history', str(plan['history_seconds']), '--forecast', str(plan['forecast_seconds']),
            '--patch-seconds', str(plan['step_seconds']), '--patch-stride', str(plan['step_seconds']),
            '--device', args.device, '--threads', str(args.threads),
            '--save-path', str(output/name)]
        if args.smoke:
            command += ['--epochs', '1', '--max-records', '8', '--d-model', '8',
                        '--n-heads', '2', '--batch-size', str(args.batch_size or 2)]
        else:
            if args.epochs is not None:
                command += ['--epochs', str(args.epochs)]
            if args.batch_size is not None:
                command += ['--batch-size', str(args.batch_size)]
        commands.append(command)
    if args.dry_run:
        for command in commands:
            print(shlex.join(command), flush=True)
        print('Dry run complete; no training or output files were created.', flush=True)
        return 0
    output.mkdir(parents=True)
    (output/'plan.json').write_text(json.dumps({'variant': 'dag', 'smoke_test': args.smoke,
        'datasets': plans, 'commands': commands}, ensure_ascii=False, indent=2))
    fields = ['dataset', 'status', 'step_seconds', 'history_steps', 'forecast_steps',
              'mae', 'mse', 'persistence_mae', 'persistence_mse', 'best_epoch',
              'metric_scale', 'metric_reduction', 'smoke_test', 'log', 'metrics']
    failed = []
    with (output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        for plan, command in zip(plans, commands):
            name = plan['dataset']
            log_path = output/f'{name}.log'
            metrics_path = output/name/'metrics.json'
            row = {key: plan[key] for key in ['dataset', 'step_seconds', 'history_steps', 'forecast_steps']}
            row.update(smoke_test=args.smoke, log=str(log_path), metrics=str(metrics_path))
            print(f'\nStarting {name}; log: {log_path}', flush=True)
            print(shlex.join(command), flush=True)
            with log_path.open('w') as log:
                process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end='', flush=True)
                    code = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    process.wait()
                    raise
            try:
                if code:
                    raise ValueError(f'training exited with code {code}')
                metrics = json.loads(metrics_path.read_text())
                test = metrics['test']
                if (test.get('metric_scale') != 'train_standard'
                        or test.get('metric_reduction') != 'variable_macro'):
                    raise ValueError('expected unified Z-score metrics; install evaluation fix')
                baseline = test['baselines']['persistence']
                row.update(status='ok', mae=test['mae'], mse=test['mse'],
                    persistence_mae=baseline['mae'], persistence_mse=baseline['mse'],
                    best_epoch=metrics['best_epoch'], metric_scale=test['metric_scale'],
                    metric_reduction=test['metric_reduction'])
            except (ValueError, FileNotFoundError, KeyError) as error:
                row['status'] = f'failed: {error}'
                failed.append(name)
                print(f'{name} failed: {error}; continuing with the next dataset.', flush=True)
            writer.writerow(row)
            stream.flush()
    print(f'\nSummary: {output/"summary.csv"}', flush=True)
    if failed:
        print('Failed datasets:', ', '.join(failed), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
