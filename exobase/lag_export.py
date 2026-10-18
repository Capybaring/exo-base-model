"""Export diagnostic X->Y predictive lags; these are not causal certificates."""
import csv
from pathlib import Path

import torch


def export_lags(model, loader, device, path, targets, exog, patch_seconds):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', newline='') as stream, torch.no_grad():
        writer = csv.writer(stream)
        writer.writerow(['record_id', 'source_X', 'target_Y', 'lag_bins', 'lag_seconds',
                         'correlation', 'observed_pairs', 'valid', 'selected'])
        for batch in loader:
            estimates = model.estimate_lags({k: v.to(device) for k, v in batch.items()})
            estimates = {k: v.cpu() for k, v in estimates.items()}
            for b, record in enumerate(batch['record_ids'].tolist()):
                for j, target in enumerate(targets):
                    for i, source in enumerate(exog):
                        lag = int(estimates['lag'][b, j, i])
                        writer.writerow([record, source, target, lag if lag else '',
                            lag*patch_seconds if lag else '',
                            float(estimates['correlation'][b, j, i]),
                            int(estimates['pair_count'][b, j, i]),
                            int(estimates['valid'][b, j, i]),
                            int(estimates['selected'][b, j, i])])
    return destination
