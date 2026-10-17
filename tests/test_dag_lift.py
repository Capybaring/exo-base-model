"""Behavioral checks for directed lead estimation and causal control timing."""
import json
import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from exobase.lags import estimate_leads, align_exogenous, ExogenousLeadRefiner
from exobase.lift_layers import ComplexLinear
from exobase.model import ModelConfig, build_model


def synthetic_batch(length=64, horizon=16):
    generator = torch.Generator().manual_seed(93)
    x = torch.randn(1, 1, length + horizon, generator=generator)
    y = torch.randn(1, 1, length, generator=generator)
    y[..., 3:] = 2*x[..., :length-3] + .01*torch.randn(1, 1, length-3, generator=generator)
    history = torch.cat([y, x[..., :length]], 1).transpose(1, 2).unsqueeze(2)
    future = x[..., length:].transpose(1, 2).unsqueeze(2)
    return {'observed_data': history, 'observed_mask': torch.ones_like(history, dtype=torch.bool),
        'observed_tp': torch.arange(length).view(1, length, 1, 1).expand_as(history).float(),
        'future_exog': future, 'future_exog_mask': torch.ones_like(future, dtype=torch.bool),
        'future_exog_tp': torch.arange(length, length+horizon).view(1, horizon, 1, 1).float(),
        'query_patch_indices': torch.arange(horizon).unsqueeze(0),
        'tp_to_predict': torch.arange(length, length+horizon).unsqueeze(0).float(),
        'data_to_predict': torch.zeros(1, horizon, 1),
        'mask_predicted_data': torch.ones(1, horizon, 1, dtype=torch.bool),
        'record_ids': torch.tensor([0])}


def small_config(variant='dag-lift', length=64, horizon=16):
    return ModelConfig(1, 1, length, variant=variant, future_patches=horizon,
        d_model=8, n_heads=2, n_layers=1, dropout=0, max_lag=8, min_pairs=32,
        min_correlation=.8, leader_num=1, state_num=2)


def test_recovers_pair_specific_positive_negative_delays_with_missing_values():
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(2, 2, 128, generator=generator)
    y = torch.randn(2, 2, 128, generator=generator)
    y[:, 0, 3:] = 1.7*x[:, 0, :-3]
    y[:, 1, 5:] = -.9*x[:, 1, :-5]
    xm = torch.rand(x.shape, generator=generator) > .15
    ym = torch.rand(y.shape, generator=generator) > .15
    leads = estimate_leads(y.masked_fill(~ym, float('nan')),
        x.masked_fill(~xm, float('nan')), ym, xm, 12, 32, .9, 1)
    assert leads['lag'][:, 0, 0].tolist() == [3, 3]
    assert leads['lag'][:, 1, 1].tolist() == [5, 5]
    assert (leads['correlation'][:, 0, 0] > .999).all()
    assert (leads['correlation'][:, 1, 1] < -.999).all()
    assert leads['leaders'].tolist() == [[[0], [1]], [[0], [1]]]
    assert leads['selected'].sum().item() == 4


def test_reverse_direction_and_constant_series_are_not_valid_leaders():
    generator = torch.Generator().manual_seed(7)
    y = torch.randn(1, 1, 128, generator=generator)
    x = torch.cat([torch.randn(1, 1, 4, generator=generator), y[..., :-4]], -1)
    mask = torch.ones_like(y, dtype=torch.bool)
    leads = estimate_leads(y, x, mask, mask, 12, 60, .9, 1)
    assert not leads['valid'].any()  # X follows Y; it does not lead it
    constant = estimate_leads(y, torch.ones_like(x), mask, mask, 12, 60, .3, 1)
    assert not constant['valid'].any() and not constant['lag'].any()


def test_no_circular_wrap_and_gaps_are_not_compressed():
    x = torch.randn(1, 1, 64, generator=torch.Generator().manual_seed(23))
    y = x.roll(-2, -1)
    mask = torch.ones_like(x, dtype=torch.bool)
    # A circular estimator can return +62 from wrapping. There are only two
    # overlapping observations at that shift, so it is ineligible here.
    result = estimate_leads(y, x, mask, mask, 63, 40, .99, 1)
    assert not result['valid'].any()
    batch = synthetic_batch()
    model = build_model(small_config()).eval()
    batch['observed_mask'][:, 20:25, :, :] = False
    estimate = model.estimate_lags(batch)
    assert estimate['leader_lag'].item() == 3


def test_estimation_uses_no_future_controls_or_targets():
    batch = synthetic_batch()
    model = build_model(small_config()).eval()
    before = model.estimate_lags(batch)
    batch['future_exog'].mul_(100).add_(900)
    batch['data_to_predict'].fill_(float('nan'))
    batch['mask_predicted_data'].fill_(False)
    after = model.estimate_lags(batch)
    for key in before:
        torch.testing.assert_close(before[key], after[key])
    history_only = {k: batch[k] for k in ('observed_data', 'observed_mask')}
    for key, result in model.estimate_lags(history_only).items():
        torch.testing.assert_close(before[key], result)


def test_alignment_uses_correct_historical_and_known_future_indices():
    x = torch.arange(8.).view(1, 1, 8)
    future = torch.arange(8., 13.).view(1, 1, 5)
    m = torch.ones_like(x, dtype=torch.bool)
    estimates = {'leaders': torch.tensor([[[0]]]), 'leader_lag': torch.tensor([[[3]]]),
        'leader_valid': torch.tensor([[[True]]]), 'leader_correlation': torch.ones(1, 1, 1)}
    aligned, valid = align_exogenous(x, future, m, torch.ones_like(future, dtype=torch.bool), estimates)
    expected = (torch.arange(5., 10.) - x.mean()) / (x.var(correction=0) + 1e-8).sqrt()
    torch.testing.assert_close(aligned[0, 0, 0], expected)
    assert valid.all()


def test_future_control_change_only_affects_outputs_after_estimated_delay():
    torch.manual_seed(42)
    batch = synthetic_batch()
    model = build_model(small_config()).eval()
    assert model.estimate_lags(batch)['leader_lag'].item() == 3
    before = model(batch)[0]
    batch['future_exog'][:, 5] += 8
    after = model(batch)[0]
    # Future X bin 5 first reaches Y bin 8 (zero-based).
    torch.testing.assert_close(before[:, :8], after[:, :8], rtol=0, atol=1e-6)
    assert (before[:, 8:] - after[:, 8:]).abs().max() > 1e-4


def test_dag_exogenous_prediction_is_independent_of_y_inputs():
    batch = synthetic_batch()
    model = build_model(small_config()).eval()
    hist, hm, future, fm = model.regularize(batch)
    _, _, before = model.base_prediction(hist, hm, future, fm)
    hist[:, 0] = hist[:, 0]*15 + 20
    _, _, after = model.base_prediction(hist, hm, future, fm)
    torch.testing.assert_close(before, after)


@pytest.mark.parametrize('variant', ['dag-history', 'dag', 'dag-lift'])
def test_finite_gradients_missing_masks_and_no_future_y_leakage(variant):
    batch = synthetic_batch()
    batch['observed_mask'][:, 20:22] = False
    batch['future_exog_mask'][:, 5:7] = False
    model = build_model(small_config(variant)).eval()
    before, auxiliary = model(batch)
    batch['observed_data'][~batch['observed_mask']] = float('nan')
    batch['future_exog'][~batch['future_exog_mask']] = float('nan')
    batch['data_to_predict'].fill_(float('nan'))
    batch['mask_predicted_data'].fill_(False)
    after, _ = model(batch)
    torch.testing.assert_close(before, after)
    (before.square().mean() + auxiliary).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if variant == 'dag-lift':
        assert model.refiner.mix_layer.weight_real.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.refiner.factory.parameters())


def test_no_eligible_leader_falls_back_to_history_dag():
    batch = synthetic_batch()
    batch['observed_mask'][..., 1] = False
    model = build_model(small_config()).eval()
    history, hm, future, fm = model.regularize(batch)
    expected, _, _ = model.base_prediction(history, hm, future, fm)
    actual = model(batch)[0]
    torch.testing.assert_close(actual, expected.transpose(1, 2))


def test_dag_known_future_path_matches_original_dual_branch_forward():
    batch = synthetic_batch()
    model = build_model(small_config('dag')).eval()
    history, hm, future, fm = model.regularize(batch)
    actual, auxiliary, _ = model.base_prediction(history, hm, future, fm)
    expected, expected_auxiliary = model.core(history.transpose(1, 2), future.transpose(1, 2))
    torch.testing.assert_close(actual.transpose(1, 2), expected)
    torch.testing.assert_close(auxiliary, expected_auxiliary)


def test_complex_linear_matches_native_complex_linear():
    torch.manual_seed(8)
    layer = ComplexLinear(12, 4)
    x = torch.randn(2, 3, 12, dtype=torch.complex64)
    expected = F.linear(x, torch.complex(layer.weight_real, layer.weight_imag),
                        torch.complex(layer.bias_real, layer.bias_imag))
    torch.testing.assert_close(layer(x), expected)


def test_global_refiner_matches_pinned_official_lift_on_complete_inputs():
    # Isolate official definitions from their unrelated experiment imports.
    environment = {'torch': torch, 'nn': torch.nn, 'F': F, 'math': math}
    for path, names in [('upstream/lift/LIFT.py', {'instance_norm', 'FilterFactory', 'LeadRefiner'}),
                        ('upstream/lift/lead_estimate.py', {'shifted_leader_seq'})]:
        source = ast.parse(Path(path).read_text())
        nodes = [n for n in source.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<official-lift>', 'exec'), environment)
    torch.manual_seed(77)
    b, length, horizon, targets, sources, k = 2, 64, 9, 2, 2, 2
    history = torch.randn(b, targets + sources, length)
    future = torch.randn(b, targets + sources, horizon)
    ids = torch.tensor([2, 3]).expand(b, targets + sources, k).clone()
    shifts = torch.tensor([3, 5]).expand_as(ids).clone()
    correlations = torch.tensor([.8, -.7]).expand_as(ids).clone()
    official = environment['LeadRefiner'](length, horizon, targets + sources, k, 3).eval()
    adapted = ExogenousLeadRefiner(length, horizon, targets, k, 3, mode='global').eval()
    factory_state = official.factory.state_dict()
    factory_state['basic_state'] = factory_state['basic_state'][:targets]
    adapted.factory.load_state_dict(factory_state)
    with torch.no_grad():
        adapted.temperature.copy_(official.temperature.squeeze())
        adapted.mix_layer.weight_real.copy_(official.mix_layer.weight.real)
        adapted.mix_layer.weight_imag.copy_(official.mix_layer.weight.imag)
        adapted.mix_layer.bias_real.copy_(official.mix_layer.bias.real)
        adapted.mix_layer.bias_imag.copy_(official.mix_layer.bias.imag)
    estimates = {'leaders': ids[:, :targets] - targets, 'leader_lag': shifts[:, :targets],
        'leader_valid': torch.ones(b, targets, k, dtype=torch.bool),
        'leader_correlation': correlations[:, :targets]}
    mask = torch.ones_like(history, dtype=torch.bool)
    aligned, available = align_exogenous(history[:, targets:], future[:, targets:],
        mask[:, targets:], torch.ones_like(future[:, targets:], dtype=torch.bool), estimates)
    expected = official(history, future, ids, shifts, correlations)[:, :targets]
    actual = adapted(history[:, :targets], mask[:, :targets], future[:, :targets], aligned, available, estimates)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('mode,horizon,states', [('prefix', 1, 1), ('prefix', 5, 3), ('global', 5, 3)])
def test_single_and_odd_horizon_frequency_paths(mode, horizon, states):
    config = small_config(horizon=horizon)
    config.refiner_mode, config.state_num = mode, states
    model = build_model(config).eval()
    prediction, auxiliary = model(synthetic_batch(horizon=horizon))
    (prediction.square().mean() + auxiliary).backward()
    assert prediction.shape == (1, horizon, 1) and torch.isfinite(prediction).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_checkpoint_reload_prediction_and_lag_export(tmp_path):
    from exobase.train import main as train
    from exobase.predict import main as predict
    generator = np.random.default_rng(91)
    t = np.arange(401, dtype=np.float64)
    x = generator.normal(size=len(t)).astype(np.float32)
    y = np.roll(x, 3)
    values = np.stack([y, x], -1)
    source = tmp_path/'series.npz'
    np.savez(source, time=t, values=values, mask=np.ones_like(values, dtype=bool),
             names=np.array(['Y', 'X']), duplicate_rows=np.array(0))
    output = tmp_path/'run'
    train(['--data-path', str(source), '--targets', 'Y', '--exog', 'X',
        '--history', '32', '--forecast', '8', '--patch-seconds', '1',
        '--variant', 'dag-lift', '--epochs', '1', '--batch-size', '2',
        '--d-model', '8', '--n-heads', '2', '--min-pairs', '12', '--max-lag', '6',
        '--state-num', '2', '--dropout', '0', '--threads', '1',
        '--device', 'cpu', '--save-path', str(output)])
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['architecture_version'] == 2 and not manifest['lag_protocol']['future_x_bypass']
    assert manifest['args']['normalization'] == 'train'
    assert (output/'test_lags.csv').exists()
    csv = tmp_path/'predictions.csv'
    lags = tmp_path/'lags.csv'
    count = predict(['--checkpoint', str(output/'best_model.pt'), '--output', str(csv),
                     '--lag-output', str(lags), '--threads', '1'])
    assert count == 16 and len(csv.read_text().splitlines()) == 17
    assert lags.read_text() == (output/'test_lags.csv').read_text()
