"""Re-evaluate a saved checkpoint with the unified masked DAG-scale metrics.

No optimization, normalization change, or checkpoint modification is performed.
Works with architecture-v2 checkpoints saved before evaluation_protocol existed.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset

from .data import load_long_table, TimeRecords
from .metrics import evaluate_irregular, fit_evaluation_statistics, evaluation_protocol
from .model import build_model, ModelConfig
from .prepare import sha256
from .train import make_collator


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-path', help='Original prepared NPZ/CSV, if moved since training')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, help='Optional inference batch size')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output', help='Defaults to metrics_unified_<split>.json next to checkpoint')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args(argv)
    if args.threads < 1 or (args.batch_size is not None and args.batch_size < 1):
        p.error('threads and batch-size must be positive')
    checkpoint = Path(args.checkpoint)
    output = Path(args.output) if args.output else checkpoint.parent/f'metrics_unified_{args.split}.json'
    if output.resolve() == checkpoint.resolve():
        p.error('Evaluation output must differ from checkpoint')
    if output.exists() and not args.overwrite:
        p.error('Output exists; choose another --output or explicitly use --overwrite')
    torch.set_num_threads(args.threads)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    manifest = saved['manifest']
    settings = SimpleNamespace(**manifest['args'])
    source = args.data_path or settings.data_path
    if output.resolve() == Path(source).resolve():
        p.error('Evaluation output must differ from dataset')
    if sha256(source) != manifest['data_sha256']:
        p.error('Dataset differs from checkpoint manifest; use the original prepared file')
    table = load_long_table(source, settings.targets, settings.exog)
    records = TimeRecords(table, settings.history, settings.forecast, len(settings.targets))
    if settings.max_records is not None:
        records.records = records.records[:settings.max_records]
    for split, ids in manifest['record_indices'].items():
        if [records.records[i][0] for i in ids] != manifest['record_ids'][split]:
            p.error('Record boundaries differ from saved splits')
    saved_protocol = manifest.get('evaluation_protocol')
    if saved_protocol is not None and saved_protocol['version'] != 1:
        p.error('Unsupported saved evaluation protocol version')
    statistics = (saved_protocol['statistics'] if saved_protocol is not None else
                  fit_evaluation_statistics(records, manifest['record_indices']['train']))
    if statistics['target_names'] != settings.targets:
        p.error('Evaluation target order differs from checkpoint')
    loader = DataLoader(Subset(records, manifest['record_indices'][args.split]),
        batch_size=args.batch_size or settings.batch_size, shuffle=False,
        collate_fn=make_collator(settings, manifest['normalization']))
    config = ModelConfig(**manifest['model_config'])
    model = build_model(config, saved['state_dict'].get('memory.patterns')).to(args.device)
    model.load_state_dict(saved['state_dict'])
    metrics = evaluate_irregular(model, loader, args.device, manifest['normalization'],
                                statistics, include_baselines=True)
    result = {'checkpoint': str(checkpoint), 'split': args.split,
        'data_sha256': manifest['data_sha256'],
        'evaluation_protocol': evaluation_protocol(statistics),
        'checkpoint_selection': {'best_epoch': saved.get('best_epoch'),
            'saved_best_validation_mse': saved.get('best_validation_mse'),
            'metric_scale': 'train_standard' if saved_protocol is not None else 'model_minmax',
            'note': 'Re-evaluation does not reselect the best epoch.'},
        args.split: metrics, 'smoke_test': manifest.get('smoke_test', False)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(f'{args.split} standard_macro MSE={metrics["mse"]:.9g} '
          f'MAE={metrics["mae"]:.9g} RMSE={metrics["rmse"]:.9g}')
    print('Observed counts:', metrics['observed_count_by_target'])
    print('Persistence standard_macro:', metrics['baselines']['persistence']['by_scale']['standard']['macro'])
    print('Full metrics:', output)
    return result


if __name__ == '__main__':
    main()
