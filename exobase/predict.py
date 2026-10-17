"""Reload a checkpoint and export predictions at real query timestamps."""
import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import load_long_table, TimeRecords
from .model import ExogenousTPatchGNN, ModelConfig
from .prepare import sha256
from .train import make_collator


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-path', help='Same dataset as training; defaults to the saved path')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output', required=True)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args(argv)
    if args.threads < 1:
        p.error('threads must be positive')
    if Path(args.output).exists() and not args.overwrite:
        p.error('Output exists; choose another path or use --overwrite')
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    manifest = saved['manifest']
    settings = SimpleNamespace(**manifest['args'])
    source = args.data_path or settings.data_path
    if sha256(source) != manifest['data_sha256']:
        p.error('Dataset differs from the checkpoint manifest; use the original prepared file')
    table = load_long_table(source, settings.targets, settings.exog)
    records = TimeRecords(table, settings.history, settings.forecast, len(settings.targets))
    if settings.max_records is not None:
        records.records = records.records[:settings.max_records]
    ids = manifest['record_indices'][args.split]
    if [records.records[i][0] for i in ids] != manifest['record_ids'][args.split]:
        p.error('Record boundaries differ from the saved split')
    loader = DataLoader(Subset(records, ids), batch_size=settings.batch_size,
        collate_fn=make_collator(settings, manifest['normalization']))
    config = ModelConfig(**manifest['model_config'])
    patterns = saved['state_dict'].get('memory.patterns')
    model = ExogenousTPatchGNN(config, patterns).to(args.device)
    model.load_state_dict(saved['state_dict']); model.eval()
    n = config.target_dim
    low = np.asarray(manifest['normalization']['data_min'])[:n]
    scale = np.asarray(manifest['normalization']['data_max'])[:n]-low
    scale = scale+(scale == 0)*1e-8
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open('w', newline='') as output, torch.no_grad():
        writer = csv.writer(output)
        writer.writerow(['record_id', 'time_seconds', 'variable', 'prediction', 'observed', 'actual'])
        for batch in loader:
            prediction, _ = model({k: v.to(args.device) for k, v in batch.items()})
            prediction = prediction.cpu().numpy()*scale+low
            actual = batch['data_to_predict'].numpy()*scale+low
            mask = batch['mask_predicted_data'].numpy()
            for b, rid in enumerate(batch['record_ids'].tolist()):
                for q in range(int(batch['query_lengths'][b])):
                    stamp = float(batch['query_time_seconds'][b,q])+rid*records.duration
                    for j, name in enumerate(settings.targets):
                        writer.writerow([rid, stamp, name, float(prediction[b,q,j]),
                            int(mask[b,q,j]), float(actual[b,q,j]) if mask[b,q,j] else ''])
                        count += 1
    print(f'Exported {count} predictions to {destination}')
    return count


def cli():
    main()


if __name__ == '__main__':
    main()
