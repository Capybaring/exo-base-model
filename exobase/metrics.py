"""t-PatchGNN evaluation.py compute_error/evaluation conventions.

First average valid errors within each variable, then average available
variables. Evaluation accumulates sums/counts across ALL batches first.
MAPE intentionally matches upstream's signed normalized-target denominator.
"""
import torch


def error_sums(prediction, target, mask, kind='mse'):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('Prediction, target and mask must share [B,Q,D]')
    mask = mask.bool()
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError('Nonfinite observed target or prediction')
    truth = torch.where(mask, target, torch.zeros_like(target))
    pred = torch.where(mask, prediction, torch.zeros_like(prediction))
    if kind == 'mse':
        error = (truth-pred).square()
    elif kind == 'mae':
        error = (truth-pred).abs()
    elif kind == 'mape':
        mask = mask & truth.ne(0)
        divisor = truth + truth.eq(0).to(truth.dtype)*1e-8
        error = (truth-pred).abs()/divisor
    else:
        raise ValueError('Unknown error: ' + kind)
    return torch.where(mask, error, torch.zeros_like(error)).sum((0, 1)), mask.sum((0, 1))


def variable_mean(sums, counts):
    available = counts > 0
    if not available.any():
        return None
    return (sums[available]/(counts[available]+1e-8)).mean()


def masked_mse(prediction, target, mask):
    return variable_mean(*error_sums(prediction, target, mask, 'mse'))


@torch.no_grad()
def evaluate_irregular(model, loader, device):
    model.eval()
    totals = {}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        prediction, _ = model(batch)
        for kind in ('mse', 'mae', 'mape'):
            sums, counts = error_sums(prediction, batch['data_to_predict'], batch['mask_predicted_data'], kind)
            if kind not in totals:
                totals[kind] = [sums.double(), counts]
            else:
                totals[kind][0] += sums.double()
                totals[kind][1] += counts
    if not totals:
        raise ValueError('Empty evaluation loader')
    results = {}
    for kind, (sums, counts) in totals.items():
        value = variable_mean(sums, counts)
        name = 'mape_official_signed' if kind == 'mape' else kind
        results[name] = float(value) if value is not None else None
    if results['mse'] is None:
        raise ValueError('No observed evaluation targets')
    results.update(loss=results['mse'], rmse=results['mse']**.5,
                   observed_count_by_target=totals['mse'][1].cpu().tolist())
    return results
