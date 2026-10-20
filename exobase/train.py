"""Train Y temporal baselines, patch/LIFT, and retained DAG comparisons."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import (DIST_TARGETS, DIST_EXOG, load_long_table, TimeRecords,
    split_records, normalization_from_seen, IrregularCollator)
from .metrics import (masked_mse, evaluate_irregular, fit_evaluation_statistics,
    evaluation_protocol)
from .model import ModelConfig, build_model
from .prepare import sha256


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', help='JSON preset; explicit command-line values override it')
    p.add_argument('--task-protocol', default='custom',
        help='Experiment task label recorded in the manifest; does not imply full paper reproduction')
    p.add_argument('--data-path')
    p.add_argument('--targets', nargs='+', default=list(DIST_TARGETS))
    p.add_argument('--exog', nargs='+', default=list(DIST_EXOG))
    p.add_argument('--history', type=float, default=14400, help='History duration, seconds')
    p.add_argument('--forecast', type=float, default=7200, help='Future duration, seconds')
    p.add_argument('--patch-seconds', type=float, default=60)
    p.add_argument('--patch-stride', type=float, default=None, help='Seconds; defaults to patch width')
    p.add_argument('--variant', choices=['y-base', 'patch-base', 'patch-lift',
        'dag-history', 'dag', 'dag-lift', 'tpatch-y', 'tpatch', 'exo', 'delay'], default='patch-lift')
    p.add_argument('--no-lift', action='store_true',
        help='patch-lift -> patch-base; legacy dag-lift -> dag-history')
    p.add_argument('--patch-encoder', choices=['ttcn', 'mean'], default='ttcn',
        help='Full TTCN tokens; y-base always uses fixed-bin means; legacy DAG also supports mean')
    p.add_argument('--split', choices=['official-random', 'chronological'], default='chronological')
    p.add_argument('--normalization', choices=['seen', 'train'], default='train')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--te-dim', type=int, default=10)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--n-layers', type=int, default=1)
    p.add_argument('--tf-layers', type=int, default=1)
    p.add_argument('--node-dim', type=int, default=10)
    p.add_argument('--graph-hops', type=int, default=2)
    p.add_argument('--alpha', type=float, default=.5, help='DAG temporal/covariate mixture')
    p.add_argument('--beta', type=float, default=.1, help='Legacy DAG auxiliary loss weight; Y/patch variants use zero')
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--lag-estimator', choices=['fft', 'pearson'], default='fft',
        help='Official circular FFT/local peaks (default), or legacy masked noncircular Pearson')
    p.add_argument('--max-lag', type=int, default=48, help='Pearson only: maximum positive lag in patch bins')
    p.add_argument('--min-pairs', type=int, default=8, help='Pearson only: minimum observed pairs per lag')
    p.add_argument('--min-correlation', type=float, default=.3, help='Pearson only: minimum absolute correlation')
    p.add_argument('--leader-num', type=int, default=3)
    p.add_argument('--state-num', type=int, default=8)
    p.add_argument('--refiner-mode', choices=['prefix', 'global'], default='global',
        help='prefix bounds control availability per prediction bin; global uses horizon-wide mixing')
    p.add_argument('--delay-context', type=int, default=4)
    p.add_argument('--n-patterns', type=int, default=16)
    p.add_argument('--max-pattern-windows', type=int, default=20000)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--max-records', type=int, help='Smoke test only: restrict before splitting/scaling')
    p.add_argument('--save-path', default='results/run')
    p.add_argument('--overwrite', action='store_true')
    return p


def make_collator(args, normalization):
    return IrregularCollator(args.history, args.forecast, args.patch_seconds,
        len(args.targets), normalization, seed=args.seed, patch_stride=args.patch_stride)


def main(argv=None):
    p = parser()
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.config:
        settings = json.loads(Path(preliminary.config).read_text())
        unknown = set(settings)-{a.dest for a in p._actions}
        if unknown:
            p.error(f'Unknown configuration settings: {sorted(unknown)}')
        p.set_defaults(**settings)
    args = p.parse_args(argv)
    if args.no_lift:
        without = {'patch-lift': 'patch-base', 'patch-base': 'patch-base',
                   'dag-lift': 'dag-history', 'dag-history': 'dag-history'}
        if args.variant not in without:
            p.error('--no-lift applies to patch-lift/patch-base or legacy dag-lift/dag-history')
        args.variant = without[args.variant]
    simple = args.variant in ('y-base', 'patch-base', 'patch-lift')
    if simple:
        args.beta = 0.
        if args.variant == 'y-base':
            args.patch_encoder = 'mean'
        elif args.patch_encoder != 'ttcn':
            p.error('patch-base/patch-lift require --patch-encoder ttcn')
    if not args.data_path:
        p.error('--data-path or a preset containing data_path is required')
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.lr <= 0:
        p.error('Training counts and learning rate must be positive')
    args.patch_stride = args.patch_seconds if args.patch_stride is None else args.patch_stride
    if (args.variant.startswith('dag') or simple) and (args.patch_stride != args.patch_seconds
            or not all(math.isclose(d/args.patch_seconds, round(d/args.patch_seconds))
                       for d in (args.history, args.forecast))):
        p.error('Temporal/DAG/LIFT requires nonoverlapping patches and history/forecast aligned to the patch grid')
    if (args.variant in ('dag-lift', 'patch-lift') and args.lag_estimator == 'pearson'
            and round(args.history/args.patch_seconds) < args.min_pairs + 1):
        p.error('History must contain at least min-pairs+1 patch bins for a positive lag')
    if (args.variant in ('dag-lift', 'patch-lift') and args.lag_estimator == 'fft'
            and round(args.history/args.patch_seconds) < 3):
        p.error('Official FFT local peaks require at least 3 history patch bins')
    if args.variant == 'delay' and (args.patch_stride != args.patch_seconds
            or not math.isclose(args.history/args.patch_seconds, round(args.history/args.patch_seconds))):
        p.error('Patch-level DFT baseline requires nonoverlapping patches and an aligned history boundary')
    output = Path(args.save_path)
    if not args.overwrite and (output/'best_model.pt').exists():
        p.error('Checkpoint exists; choose a new --save-path or explicitly use --overwrite')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(args.threads)
    print('Loading:', args.data_path, flush=True)
    table = load_long_table(args.data_path, args.targets, args.exog)
    records = TimeRecords(table, args.history, args.forecast, len(args.targets))
    full_count = len(records)
    if args.max_records is not None:
        if args.max_records < 5:
            p.error('max-records must be at least five')
        records.records = records.records[:args.max_records]
    splits = split_records(records, args.split)
    scale_ids = splits['train']+(splits['val'] if args.normalization == 'seen' else [])
    norm = normalization_from_seen(records, scale_ids)
    # Evaluation scaling is independent of model scaling, as in DAG/TFB.
    # Even --normalization seen must not use validation targets for this scaler.
    eval_statistics = fit_evaluation_statistics(records, splits['train'])
    loaders = {name: DataLoader(Subset(records, indices), batch_size=args.batch_size,
        shuffle=name == 'train', drop_last=False, num_workers=0,
        collate_fn=make_collator(args, norm), generator=torch.Generator().manual_seed(args.seed))
        for name, indices in splits.items()}
    config = ModelConfig(target_dim=len(args.targets), exog_dim=len(args.exog),
        history_patches=math.ceil(max(args.history-args.patch_seconds, 0)/args.patch_stride)+1,
        variant=args.variant, d_model=args.d_model, te_dim=args.te_dim,
        n_heads=args.n_heads, n_layers=args.n_layers, tf_layers=args.tf_layers,
        node_dim=args.node_dim, graph_hops=args.graph_hops, delay_context=args.delay_context,
        future_patches=math.ceil(args.forecast/args.patch_seconds), alpha=args.alpha,
        beta=args.beta, dropout=args.dropout, max_lag=args.max_lag, min_pairs=args.min_pairs,
        min_correlation=args.min_correlation, leader_num=args.leader_num,
        state_num=args.state_num, refiner_mode=args.refiner_mode, patch_encoder=args.patch_encoder,
        lag_estimator=args.lag_estimator)
    patterns, pattern_info = None, None
    if args.variant == 'delay':
        from .patterns import fit_pattern_bank
        print('Fitting train-only k-Shape pattern bank', flush=True)
        patterns, pattern_info = fit_pattern_bank(records, splits['train'], make_collator(args, norm),
            len(args.targets), args.delay_context, args.n_patterns, args.max_pattern_windows, args.seed)
        np.savez_compressed(output/'patterns.npz', patterns=patterns)
    model = build_model(config, patterns).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    manifest = {'args': vars(args), 'model_config': asdict(config), 'normalization': norm,
        'forecast_task': {'protocol': args.task_protocol, 'history_seconds': args.history,
            'forecast_seconds': args.forecast, 'patch_seconds': args.patch_seconds,
            'history_patches': config.history_patches, 'future_patches': config.future_patches,
            'window_alignment_only': args.task_protocol in ('dataset-short', 'dataset-long'),
            'data_protocol': 'irregular observations, disjoint records; not TFB rolling',
            'record_split': args.split,
            'chronological_ratios': [0.6, 0.2, 0.2] if args.split == 'chronological' else None},
        'evaluation_protocol': evaluation_protocol(eval_statistics),
        'data_sha256': sha256(args.data_path),
        'record_ids': {name: [records.records[i][0] for i in ids] for name, ids in splits.items()},
        'record_indices': splits, 'records_before_smoke_limit': full_count,
        'skipped_empty_records': records.skipped, 'duplicates_merged': table.duplicate_rows,
        'table_shape': list(table.values.shape), 'variable_order_y_then_x': list(table.names),
        'incomplete_tail_seconds': float(table.time[-1] % records.duration),
        'normalization_scope': 'train+validation (official seen_data)' if args.normalization == 'seen' else 'train only',
        'known_future_x': args.variant in ('patch-lift', 'dag', 'dag-lift', 'exo', 'delay'), 'pattern_bank': pattern_info,
        'architecture_version': 4 if simple else (3 if args.variant.startswith('dag') and args.patch_encoder == 'ttcn' else 2),
        'backbone': (('DAG dual attention on full TTCN tokens' if args.patch_encoder == 'ttcn'
                      else 'DAG dual attention on masked patch means')
                     if args.variant.startswith('dag') else 'legacy t-PatchGNN'),
        'patch_protocol': ({'encoder': 'official t-PatchGNN point/time TTCN, safe empty patches',
            'features': args.d_model, 'scalar_mean_before_backbone': False,
            'future_x_encoder': 'shared with history, per patch only',
            'decoder': 'forecast token + actual query-time embedding',
            'auxiliary_targets': 'raw observed future X and historical Y, not learned token labels'}
            if args.variant.startswith('dag') and args.patch_encoder == 'ttcn' else None),
        'lag_protocol': ({'direction': 'X->Y only',
            'estimator': ('official circular FFT/local peaks on history, shared-feature spectral mean'
                          if args.lag_estimator == 'fft' else 'history-only masked noncircular Pearson'),
            'feature_space': 'TTCN tokens' if args.patch_encoder == 'ttcn' else 'scalar patch means',
            'lag_estimator': args.lag_estimator,
            'positive_lags_only': True,
            'max_lag_seconds': ((config.history_patches-2) if args.lag_estimator == 'fft'
                                else min(args.max_lag, config.history_patches-1))*args.patch_seconds,
            'min_pairs': args.min_pairs if args.lag_estimator == 'pearson' else None,
            'min_correlation': args.min_correlation if args.lag_estimator == 'pearson' else None,
            'pair_count': 'circular observed pairs' if args.lag_estimator == 'fft' else 'noncircular observed pairs',
            'leader_stream_detached': True,
            'refiner_mode': args.refiner_mode, 'future_x_bypass': False,
            'learned_lag_estimator': False} if args.variant == 'dag-lift' else None),
        'smoke_test': args.max_records is not None,
        'sources': {'tpatchgnn': '00c94e7bbaf21c71b03ed84ff690ae59e37129e5',
                    'dag_data_adapter': 'f4468901331f53bb84bbc405219150df997e26c5',
                    'dag_model': 'f4468901331f53bb84bbc405219150df997e26c5',
                    'lift': '79cf8f157bb9a616c9733daf1e9c2cfefdc98c94',
                    'pdformer': 'f8c8f6ad007a04fad3baee958b89504711852ce9'}}
    if simple:
        manifest['backbone'] = 'Y-only shared temporal attention, compact direct multi-step head; no DAG transfer'
        manifest['evaluation_protocol']['training_objective'] = 'MinMax masked variable-macro MSE on future Y only'
        manifest['patch_protocol'] = {
            'encoder': ('fixed-bin observed scalar means, no TTCN' if args.variant == 'y-base'
                        else 'unchanged point/time TTCN, shared within each X/Y role'),
            'features': args.d_model, 'scalar_mean_before_backbone': args.variant == 'y-base',
            'separate_x_y_encoders': args.variant == 'patch-lift',
            'future_x_encoder': 'shared with historical X' if args.variant == 'patch-lift' else None,
            'decoder': 'forecast Y token + actual query-time embedding from Y encoder',
            'auxiliary_targets': None}
        if args.variant == 'patch-lift':
            manifest['lag_protocol'] = {
                'direction': 'X->Y only', 'lag_estimator': args.lag_estimator,
                'estimator': ('circular FFT/local peaks' if args.lag_estimator == 'fft'
                              else 'masked noncircular Pearson'),
                'feature_space': 'observed scalar patch means; lag diagnostics only',
                'scalar_mean_before_backbone': False,
                'positive_lags_only': True,
                'max_lag_seconds': ((config.history_patches-2) if args.lag_estimator == 'fft'
                    else min(args.max_lag, config.history_patches-1))*args.patch_seconds,
                'min_pairs': args.min_pairs if args.lag_estimator == 'pearson' else None,
                'min_correlation': args.min_correlation if args.lag_estimator == 'pearson' else None,
                'pair_count': 'circular observed pairs' if args.lag_estimator == 'fft' else 'noncircular observed pairs',
                'leader_stream_detached': False, 'feature_mapping': 'learned normalized X->Y linear map',
                'refiner_mode': args.refiner_mode, 'future_x_bypass': False,
                'learned_lag_estimator': False}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('Variant:', args.variant, 'records:', {k: len(v) for k, v in splits.items()},
          'parameters:', sum(x.numel() for x in model.parameters()), flush=True)
    print('Evaluation: train-only Z-score, observed targets, variable macro; '
          'training objective: '+('future-Y MinMax MSE only' if simple else 'MinMax MSE + auxiliary'), flush=True)
    best, best_epoch, best_state, stale, history = float('inf'), 0, None, 0, []
    for epoch in range(args.epochs):
        model.train(); start = time.perf_counter(); steps = 0
        for index, batch in enumerate(loaders['train']):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            optimizer.zero_grad()
            prediction, auxiliary = model(batch)
            loss = masked_mse(prediction, batch['data_to_predict'], batch['mask_predicted_data'])
            if loss is None:
                continue
            total = loss+auxiliary
            if not torch.isfinite(total):
                raise ValueError('Nonfinite training loss')
            total.backward(); optimizer.step(); steps += 1
            if index == 0 or (index+1) % 50 == 0:
                print(f'Epoch {epoch+1} batch {index+1}/{len(loaders["train"])} '
                      f'train_minmax_mse={float(loss.detach()):.6f} '
                      f'auxiliary={float(auxiliary.detach()):.6f} '
                      f'total={float(total.detach()):.6f}', flush=True)
        if not steps:
            raise ValueError('No observed training targets')
        val = evaluate_irregular(model, loaders['val'], args.device, norm, eval_statistics)
        history.append({'epoch': epoch+1, 'validation': val})
        print(f'Epoch {epoch+1} val standard_macro_MSE={val["mse"]:.6f} '
              f'MAE={val["mae"]:.6f} seconds={time.perf_counter()-start:.1f}', flush=True)
        if val['mse'] < best:
            best, best_epoch, stale = val['mse'], epoch+1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'state_dict': best_state, 'manifest': manifest, 'best_epoch': best_epoch,
                'best_validation_mse': best}, output/'best_model.pt')
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    test = evaluate_irregular(model, loaders['test'], args.device, norm, eval_statistics,
                              include_baselines=True)
    results = {'best_epoch': best_epoch, 'best_validation_mse': best, 'test': test,
        'evaluation_protocol': manifest['evaluation_protocol'],
        'training_history': history, 'smoke_test': args.max_records is not None}
    (output/'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    if args.variant in ('dag-lift', 'patch-lift'):
        from .lag_export import export_lags
        export_lags(model, loaders['test'], args.device, output/'test_lags.csv',
                    args.targets, args.exog, args.patch_seconds)
    print('Test:', json.dumps(test), flush=True)
    return results


def cli():
    main()


if __name__ == '__main__':
    main()
