"""Masked forecasting metrics, with an explicit scale and reduction protocol.

Model inputs/losses remain MinMax for checkpoint compatibility. New evaluations
undo that transform per target, then divide errors by train-only population
standard deviations (DAG/TFB mse_norm convention). The primary reduction is
t-PatchGNN's variable macro average; point micro averages are also exported.
Calling evaluate_irregular without scale metadata retains the legacy API.
"""
import numpy as np
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


def fit_evaluation_statistics(records, train_indices):
    """Observed training cells only, before patch aggregation; ddof=0.

    Pairwise variance merging avoids cancellation on large-offset variables.
    TimeRecords are disjoint, so no training observation is counted twice.
    """
    dim = records.target_dim
    count = np.zeros(dim, dtype=np.int64)
    mean, m2 = np.zeros(dim), np.zeros(dim)
    if len(set(train_indices)) != len(train_indices):
        raise ValueError('Duplicate training indices in evaluation statistics')
    for index in train_indices:
        _, _, values, mask = records[index]
        for j in range(dim):
            observed = values[mask[:, j], j].astype(np.float64)
            if not len(observed):
                continue
            if not np.isfinite(observed).all():
                raise ValueError('Nonfinite observed training target')
            n = len(observed)
            local_mean = observed.mean()
            delta = local_mean - mean[j]
            total = count[j] + n
            m2[j] += ((observed-local_mean)**2).sum() + delta**2*count[j]*n/total
            mean[j] += delta*n/total
            count[j] = total
    if not (count > 0).all():
        raise ValueError('Every target needs observed training values for evaluation scaling')
    variance = np.maximum(m2/count, 0.)
    # StandardScaler's numerical constant-feature bound (float64 arithmetic).
    eps = np.finfo(np.float64).eps
    constant = variance <= count*eps*variance + (count*mean*eps)**2
    scale = np.where(constant, 1., np.sqrt(variance))
    return {'target_names': list(records.table.names[:dim]),
            'mean': mean.tolist(), 'std': np.sqrt(variance).tolist(),
            'scale': scale.tolist(), 'observed_count': count.tolist(),
            'scope': 'training records, observed cells only, history and forecast',
            'ddof': 0}


def evaluation_protocol(statistics):
    return {'version': 1, 'primary_scale': 'standard',
            'primary_reduction': 'variable_macro',
            'mask': 'observed forecast targets only; exclude padding and missing cells',
            'accumulation': 'sum errors and counts over all batches before division',
            'statistics': statistics,
            'model_input_scale': 'minmax (saved checkpoint normalization)',
            'training_objective': 'MinMax masked variable-macro MSE plus model auxiliary loss',
            'rmse': 'sqrt of final MSE, not average of window or variable RMSE'}


class ForecastAccumulator:
    """Accumulate per-target errors in float64; no averaging of batch means."""

    def __init__(self, normalization, statistics):
        dim = len(statistics['scale'])
        low = np.asarray(normalization['data_min'], dtype=np.float64)[:dim]
        span = np.asarray(normalization['data_max'], dtype=np.float64)[:dim] - low
        span = span + (span == 0)*1e-8  # exactly IrregularCollator's inverse
        std = np.asarray(statistics['scale'], dtype=np.float64)
        if (len(span) != dim or not np.isfinite(span).all() or (span <= 0).any()
                or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError('Invalid per-target evaluation scale')
        self.factors = {'standard': span/std, 'raw': span, 'model_minmax': np.ones(dim)}
        self.statistics = statistics
        self.counts = np.zeros(dim, dtype=np.int64)
        self.sums = {name: {'mse': np.zeros(dim), 'mae': np.zeros(dim)}
                     for name in self.factors}

    def update(self, prediction, target, mask):
        if prediction.shape[-1] != len(self.counts):
            raise ValueError('Prediction target dimension differs from evaluation statistics')
        # Convert before subtraction/squaring, not after a float32 batch sum.
        prediction, target = prediction.double(), target.double()
        mse, count = error_sums(prediction, target, mask, 'mse')
        mae, _ = error_sums(prediction, target, mask, 'mae')
        self.counts += count.cpu().numpy()
        mse, mae = mse.cpu().numpy(), mae.cpu().numpy()
        for name, factor in self.factors.items():
            self.sums[name]['mse'] += mse*factor**2
            self.sums[name]['mae'] += mae*factor

    def result(self):
        valid = self.counts > 0
        if not valid.any():
            raise ValueError('No observed evaluation targets')
        by_scale = {}
        for name, sums in self.sums.items():
            entry = {'macro': {}, 'micro': {}, 'per_target': {}}
            for kind in ('mse', 'mae'):
                per_target = np.divide(sums[kind], self.counts,
                    out=np.zeros_like(sums[kind]), where=valid)
                entry['macro'][kind] = float(per_target[valid].mean())
                entry['micro'][kind] = float(sums[kind].sum()/self.counts.sum())
                entry['per_target'][kind] = [float(v) if ok else None
                                             for v, ok in zip(per_target, valid)]
            for reduction in ('macro', 'micro'):
                entry[reduction]['rmse'] = entry[reduction]['mse']**.5
            entry['per_target']['rmse'] = [v**.5 if v is not None else None
                                           for v in entry['per_target']['mse']]
            by_scale[name] = entry
        primary = by_scale['standard']['macro']
        return {**primary, 'loss': primary['mse'],
                'observed_count_by_target': self.counts.tolist(),
                'target_names': self.statistics['target_names'],
                'by_scale': by_scale,
                'metric_scale': 'train_standard', 'metric_reduction': 'variable_macro'}


def historical_baselines(batch, normalization, statistics):
    """Last observed Y and train-mean forecasts on exactly the same query mask.

    Pick by actual time, not padded slot or patch index. If a target has no
    history in a record, persistence falls back to its observed training mean.
    No future Y or future X enters either baseline.
    """
    truth = batch['data_to_predict']
    dim = truth.shape[-1]
    low = truth.new_tensor(normalization['data_min'][:dim])
    span = truth.new_tensor(normalization['data_max'][:dim])-low
    span = span + span.eq(0)*1e-8
    constant = (truth.new_tensor(statistics['mean'])-low)/span
    train_mean = constant.view(1, 1, dim).expand_as(truth)
    data = batch['observed_data'][..., :dim].flatten(1, 2)
    valid = batch['observed_mask'][..., :dim].flatten(1, 2).bool()
    time = batch['observed_tp'][..., :dim].flatten(1, 2)
    index = time.masked_fill(~valid, -torch.inf).argmax(1, keepdim=True)
    last = data.gather(1, index).squeeze(1)
    available = valid.any(1)
    last = torch.where(available, last, constant)
    return {'persistence': last[:, None, :].expand_as(truth), 'train_mean': train_mean}, \
           (~available).sum(0)


@torch.no_grad()
def evaluate_irregular(model, loader, device, normalization=None, statistics=None,
                       include_baselines=False):
    if (normalization is None) != (statistics is None):
        raise ValueError('Provide both checkpoint normalization and train statistics')
    model.eval()
    if statistics is not None:
        accumulator = ForecastAccumulator(normalization, statistics)
        baseline_totals = {name: ForecastAccumulator(normalization, statistics)
                           for name in ('persistence', 'train_mean')} if include_baselines else {}
        fallback = np.zeros(len(statistics['scale']), dtype=np.int64)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            prediction, _ = model(batch)
            truth, mask = batch['data_to_predict'], batch['mask_predicted_data']
            accumulator.update(prediction, truth, mask)
            if include_baselines:
                baselines, missing = historical_baselines(batch, normalization, statistics)
                fallback += missing.cpu().numpy()
                for name, estimate in baselines.items():
                    baseline_totals[name].update(estimate, truth, mask)
        result = accumulator.result()
        result['baselines'] = {name: total.result() for name, total in baseline_totals.items()}
        if include_baselines:
            result['baselines']['persistence']['train_mean_fallback_records_by_target'] = fallback.tolist()
        return result
    # Legacy callers keep the official t-PatchGNN MinMax metric names/MAPE.
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
