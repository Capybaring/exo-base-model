"""DAG-scale parity, irregular reduction and legacy checkpoint integration."""
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from exobase.data import ObservationTable, TimeRecords
from exobase.metrics import (ForecastAccumulator, fit_evaluation_statistics,
    historical_baselines)


def metadata(dim=2):
    return ({'data_min': [0.]*dim, 'data_max': [1.]*dim},
            {'target_names': [f'Y{i}' for i in range(dim)],
             'mean': [0.]*dim, 'scale': [1.]*dim})


def test_dense_standard_metrics_equal_dag_and_rolling_window_mse_mae():
    # Pinned DAG formulas, exercised using sklearn's actual scaler.
    generator = np.random.default_rng(21)
    train = generator.normal(size=(127, 2))*[3., 100.] + [20., -600.]
    scaler = StandardScaler().fit(train)
    low, high = train.min(0), train.max(0)
    truth = generator.normal(size=(3, 7, 2))*[3., 100.] + [20., -600.]
    prediction = truth + generator.normal(size=truth.shape)*[.8, 27.]
    normalization = {'data_min': low.tolist(), 'data_max': high.tolist()}
    statistics = {'target_names': ['A', 'B'], 'mean': scaler.mean_.tolist(),
                  'scale': scaler.scale_.tolist()}
    total = ForecastAccumulator(normalization, statistics)
    p = torch.tensor((prediction-low)/(high-low))
    y = torch.tensor((truth-low)/(high-low))
    total.update(p, y, torch.ones_like(y, dtype=torch.bool))
    result = total.result()
    zerror = (prediction-truth)/scaler.scale_
    expected_mse = float((zerror**2).mean())
    expected_mae = float(np.abs(zerror).mean())
    assert result['mse'] == pytest.approx(expected_mse)
    assert result['mae'] == pytest.approx(expected_mae)
    assert result['mse'] == pytest.approx(np.mean([(e**2).mean() for e in zerror]))
    assert result['mae'] == pytest.approx(np.mean([np.abs(e).mean() for e in zerror]))
    assert result['by_scale']['standard']['micro']['mse'] == pytest.approx(expected_mse)
    assert result['by_scale']['raw']['macro']['mse'] == pytest.approx(((prediction-truth)**2).mean())
    assert result['rmse'] == pytest.approx(expected_mse**.5)
    # Mean window RMSE is generally different, even on fully observed data.
    assert result['rmse'] != pytest.approx(np.mean([np.sqrt((e**2).mean()) for e in zerror]))


def test_irregular_macro_and_micro_diverge_and_ignore_missing_nan():
    normalization, statistics = metadata()
    p, y = torch.zeros(1, 9, 2), torch.zeros(1, 9, 2)
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[:, 1:, 1] = False
    p[0, 0, 1] = 10.
    p[~mask] = float('nan')
    y[~mask] = float('inf')
    total = ForecastAccumulator(normalization, statistics)
    total.update(p, y, mask)
    result = total.result()
    assert result['mse'] == 50.
    assert result['by_scale']['standard']['micro']['mse'] == 10.
    assert result['observed_count_by_target'] == [9, 1]


@pytest.mark.parametrize('dim', [1, 3])
def test_batch_partition_padding_and_empty_variable_invariance(dim):
    normalization, statistics = metadata(dim)
    generator = torch.Generator().manual_seed(42)
    y = torch.randn(5, 7, dim, generator=generator)
    p = y + torch.randn(y.shape, generator=generator)
    mask = torch.rand(y.shape, generator=generator) > .3
    if dim > 1:
        mask[..., -1] = False
    full, split = [ForecastAccumulator(normalization, statistics) for _ in range(2)]
    full.update(p, y, mask)
    for start, stop in [(0, 2), (2, 4), (4, 5)]:
        split.update(p[start:stop], y[start:stop], mask[start:stop])
    expected, actual = full.result(), split.result()
    assert actual['mse'] == pytest.approx(expected['mse'], abs=1e-14)
    assert actual['mae'] == pytest.approx(expected['mae'], abs=1e-14)
    assert actual['observed_count_by_target'] == expected['observed_count_by_target']
    padding = torch.full((5, 11, dim), float('nan'))
    full.update(padding, padding, torch.zeros_like(padding, dtype=torch.bool))
    assert full.result() == expected
    if dim == 1:
        assert actual['mse'] == actual['by_scale']['standard']['micro']['mse']
    else:
        assert actual['by_scale']['standard']['per_target']['mse'][-1] is None


def test_convert_each_target_before_averaging():
    normalization, statistics = metadata()
    normalization['data_max'] = [2., 100.]
    statistics['scale'] = [2., 10.]
    total = ForecastAccumulator(normalization, statistics)
    p = torch.tensor([[[1., 2.]]])
    total.update(p, torch.zeros_like(p), torch.ones_like(p, dtype=torch.bool))
    result = total.result()
    assert result['mse'] == 200.5  # (1 * 1**2 + 4 * 10**2)/2
    assert result['mse'] != result['by_scale']['model_minmax']['macro']['mse'] * (1+100)/2


def test_train_statistics_match_standard_scaler_and_exclude_missing_validation_test():
    time = np.arange(61, dtype=np.float64)
    values = np.stack([time*2+1e6, np.full_like(time, 7.), time], -1)
    mask = np.ones_like(values, dtype=bool)
    mask[::3, 0] = False
    values[~mask] = 1e100
    table = ObservationTable(time, values, mask, ('Y1', 'Y2', 'X'), 0)
    records = TimeRecords(table, 4, 2, 2)
    indices = [0, 1, 2]
    stats = fit_evaluation_statistics(records, indices)
    train_rows = np.r_[0:18]
    train = np.where(mask[train_rows, :2], values[train_rows, :2], np.nan)
    scaler = StandardScaler().fit(train)
    np.testing.assert_allclose(stats['mean'], scaler.mean_)
    np.testing.assert_allclose(stats['scale'], scaler.scale_)
    assert stats['scale'][1] == 1. and stats['std'][1] == 0.
    values[18:] = 9e20
    assert fit_evaluation_statistics(records, indices) == stats
    with pytest.raises(ValueError, match='Duplicate'):
        fit_evaluation_statistics(records, [0, 0])


def test_persistence_uses_actual_time_and_no_future_labels():
    normalization, stats = metadata()
    stats['mean'] = [20., 30.]
    batch = {'data_to_predict': torch.zeros(1, 5, 2),
             'observed_data': torch.tensor([[[[3., float('nan')], [9., float('nan')]],
                                             [[4., float('nan')], [7., float('nan')]]]]),
             'observed_tp': torch.tensor([[[[.1, .1], [.9, .9]], [[.4, .4], [.2, .2]]]]),
             'observed_mask': torch.tensor([[[[1, 0], [1, 0]], [[1, 0], [1, 0]]]], dtype=torch.bool)}
    forecasts, missing = historical_baselines(batch, normalization, stats)
    assert forecasts['persistence'][0, :, 0].tolist() == [9.]*5
    assert forecasts['persistence'][0, :, 1].tolist() == [30.]*5
    assert missing.tolist() == [0, 1]
    batch['data_to_predict'].fill_(float('nan'))
    again, _ = historical_baselines(batch, normalization, stats)
    torch.testing.assert_close(again['persistence'], forecasts['persistence'])


def test_new_training_and_old_checkpoint_reevaluation(tmp_path):
    from exobase.train import main as train
    from exobase.evaluate import main as evaluate
    from exobase.predict import main as predict
    generator = np.random.default_rng(10)
    time = np.arange(401, dtype=np.float64)
    values = generator.normal(size=(401, 2)).astype(np.float32)*[4., 12.] + [80., 500.]
    mask = generator.random(size=values.shape) > .25
    source = tmp_path/'series.npz'
    np.savez(source, time=time, values=values, mask=mask,
             names=np.array(['Y', 'X']), duplicate_rows=np.array(0))
    output = tmp_path/'run'
    original = train(['--data-path', str(source), '--targets', 'Y', '--exog', 'X',
        '--history', '32', '--forecast', '8', '--patch-seconds', '1',
        '--variant', 'dag-history', '--epochs', '1', '--batch-size', '2',
        '--d-model', '8', '--n-heads', '2', '--dropout', '0', '--threads', '1',
        '--device', 'cpu', '--save-path', str(output)])
    checkpoint = output/'best_model.pt'
    before_bytes = checkpoint.read_bytes()
    result = evaluate(['--checkpoint', str(checkpoint), '--threads', '1', '--batch-size', '1'])
    assert checkpoint.read_bytes() == before_bytes
    assert result['test']['mse'] == pytest.approx(original['test']['mse'], rel=1e-5)
    assert result['test']['baselines']['persistence']['observed_count_by_target'] == result['test']['observed_count_by_target']
    predictions = tmp_path/'predictions.csv'
    predict(['--checkpoint', str(checkpoint), '--output', str(predictions)])
    saved = torch.load(checkpoint, weights_only=True)
    del saved['manifest']['evaluation_protocol']
    legacy = tmp_path/'legacy.pt'
    torch.save(saved, legacy)
    legacy_bytes = legacy.read_bytes()
    old = evaluate(['--checkpoint', str(legacy), '--threads', '1',
                    '--output', str(tmp_path/'legacy_metrics.json')])
    assert old['test']['mse'] == pytest.approx(original['test']['mse'])
    assert old['checkpoint_selection']['metric_scale'] == 'model_minmax'
    assert legacy.read_bytes() == legacy_bytes
    old_predictions = tmp_path/'old_predictions.csv'
    predict(['--checkpoint', str(legacy), '--output', str(old_predictions)])
    assert old_predictions.read_bytes() == predictions.read_bytes()
    assert original['evaluation_protocol']['statistics']['ddof'] == 0
    with pytest.raises(SystemExit):
        evaluate(['--checkpoint', str(legacy), '--output', str(legacy), '--overwrite'])
