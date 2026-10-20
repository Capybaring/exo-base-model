"""TTCN upstream parity and vector DAG/LIFT behavioral regression checks."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from exobase.ttcn import TTCNPatchEncoder
from exobase.model import ModelConfig, build_model
from exobase.token_lift import estimate_token_leads, align_token_exogenous, TokenLeadRefiner
from exobase.lags import ExogenousLeadRefiner


@pytest.mark.parametrize('execution', ['chunked', 'full'])
def test_ttcn_matches_official_nonempty_patch_formula_and_gradients(execution):
    source = ast.parse(Path('upstream/tpatchgnn/tPatchGNN/model/tPatchGNN.py').read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'tPatchGNN')
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('TTCN', 'LearnableTE')]
    env = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<official TTCN>', 'exec'), env)
    torch.manual_seed(19)
    encoder = TTCNPatchEncoder(8, 4, execution=execution).train()
    values = torch.randn(2, 3, 5, 2)
    times = torch.rand_like(values)
    mask = torch.rand_like(values) > .3
    mask[:, :, 0] = True
    actual, valid = encoder(values, times, mask)
    v = values.permute(0, 3, 1, 2).reshape(-1, 5, 1)
    t = times.permute(0, 3, 1, 2).reshape(-1, 5, 1)
    m = mask.permute(0, 3, 1, 2).reshape(-1, 5, 1)
    stub = SimpleNamespace(Filter_Generators=encoder.Filter_Generators,
        T_bias=encoder.T_bias, ttcn_dim=7, te_scale=encoder.te_scale, te_periodic=encoder.te_periodic)
    features = torch.cat([v, env['LearnableTE'](stub, t)], -1)
    expected = env['TTCN'](stub, features, m.float()).reshape(2, 2, 3, 7)
    torch.testing.assert_close(actual[..., :-1], expected)
    assert valid.all() and actual[..., -1].eq(1).all()
    parameters = tuple(encoder.parameters())
    official_grads = torch.autograd.grad(expected.square().sum(), parameters, retain_graph=True)
    actual_grads = torch.autograd.grad(actual[..., :-1].square().sum(), parameters, retain_graph=True)
    for actual_grad, official_grad in zip(actual_grads, official_grads):
        torch.testing.assert_close(actual_grad, official_grad)
    actual.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


@pytest.mark.parametrize('points', [1, 7, 40])
@pytest.mark.parametrize('training', [False, True])
def test_full_ttcn_matches_chunked_with_empty_patches_padding_and_input_gradients(points, training):
    torch.manual_seed(211)
    chunked = TTCNPatchEncoder(8, 4).train(training)
    full = TTCNPatchEncoder(8, 4, execution='full').train(training)
    full.load_state_dict(chunked.state_dict())
    mask = torch.rand(2, 17, points, 3) > .3
    mask[:, 2:4] = False
    mask[:, :, -1] = False  # trailing padding, including all-empty P=1
    values = torch.randn(mask.shape).masked_fill(~mask, float('nan'))
    times = torch.rand(mask.shape).masked_fill(~mask, float('nan'))
    inputs = [(values.clone().requires_grad_(), times.clone().requires_grad_()) for _ in range(2)]
    outputs = [module(v, t, mask) for module, (v, t) in zip((chunked, full), inputs)]
    torch.testing.assert_close(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])
    for output in outputs:
        output[0].square().sum().backward()
    for a, b in zip(chunked.parameters(), full.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)
    for a, b in zip(inputs[0], inputs[1]):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)
        assert torch.isfinite(b.grad).all()


@pytest.mark.parametrize('variant', ['patch-base', 'patch-lift', 'dag'])
def test_full_execution_preserves_model_predictions_gradients_and_checkpoint_config(variant):
    from dataclasses import asdict
    torch.manual_seed(119)
    settings = config(variant)
    chunked = build_model(settings).train()
    saved = asdict(settings)
    saved['ttcn_execution'] = 'full'
    full = build_model(ModelConfig.from_saved(saved)).train()
    full.load_state_dict(chunked.state_dict())
    before, aux = chunked(token_batch())
    after, full_aux = full(token_batch())
    torch.testing.assert_close(after, before)
    torch.testing.assert_close(full_aux, aux)
    (before.square().mean()+aux).backward()
    (after.square().mean()+full_aux).backward()
    for (name, a), (_, b) in zip(chunked.named_parameters(), full.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5, msg=name)
    assert ModelConfig.from_saved(saved).ttcn_execution == 'full'
    saved.pop('ttcn_execution')
    assert ModelConfig.from_saved(saved).ttcn_execution == 'chunked'


def test_ttcn_preserves_distribution_time_and_ignores_padding_and_point_order():
    torch.manual_seed(5)
    encoder = TTCNPatchEncoder(8, 4).eval()
    values = torch.tensor([[[[.1], [.5], [.9]]]])
    times = torch.tensor([[[[.01], [.04], [.09]]]])
    mask = torch.ones_like(values, dtype=torch.bool)
    before = encoder(values, times, mask)[0]
    same_mean = torch.tensor([[[[.4], [.5], [.6]]]])
    assert not torch.allclose(before, encoder(same_mean, times, mask)[0])
    assert not torch.allclose(before, encoder(values, times+.3, mask)[0])
    permutation = torch.tensor([2, 0, 1])
    torch.testing.assert_close(before, encoder(values[:, :, permutation], times[:, :, permutation], mask)[0])
    pad = torch.full((1, 1, 5, 1), float('nan'))
    padded = encoder(torch.cat([values, pad], 2), torch.cat([times, pad], 2),
                     torch.cat([mask, torch.zeros_like(pad, dtype=torch.bool)], 2))[0]
    torch.testing.assert_close(before, padded)
    empty, valid = encoder(pad, pad, torch.zeros_like(pad, dtype=torch.bool))
    assert not valid.any() and empty.eq(0).all()


def test_token_leads_recover_signed_delays_with_missing_and_no_future_fields():
    generator = torch.Generator().manual_seed(33)
    x = torch.randn(2, 2, 96, 5, generator=generator)
    y = torch.randn(2, 2, 96, 5, generator=generator)
    y[:, 0, 3:] = 2*x[:, 0, :-3]+4
    y[:, 1, 5:] = -3*x[:, 1, :-5]+7
    xm = torch.rand(2, 2, 96, generator=generator) > .15
    ym = torch.rand(2, 2, 96, generator=generator) > .15
    leads = estimate_token_leads(y, x, ym, xm, 12, 24, .99, 1, estimator="pearson")
    assert leads['lag'][:, 0, 0].tolist() == [3, 3]
    assert leads['lag'][:, 1, 1].tolist() == [5, 5]
    assert (leads['correlation'][:, 0, 0] > .999).all()
    assert (leads['correlation'][:, 1, 1] < -.999).all()
    constant = estimate_token_leads(y, torch.ones_like(x), ym, xm, 12, 24, .3, 1, estimator="pearson")
    assert not constant['valid'].any()


@pytest.mark.parametrize('mode', ['prefix', 'global'])
def test_one_feature_token_refiner_reduces_to_previous_lift_equations(mode):
    torch.manual_seed(40)
    history = torch.randn(2, 2, 16)
    mask = torch.rand_like(history) > .2
    base = torch.randn(2, 2, 7)
    aligned = torch.randn(2, 2, 2, 7)
    available = torch.rand_like(aligned) > .2
    estimates = {'leader_correlation': torch.tensor([[[.7, -.8], [.6, .9]]]).expand(2, -1, -1),
                 'leader_valid': torch.ones(2, 2, 2, dtype=torch.bool)}
    old = ExogenousLeadRefiner(16, 7, 2, 2, 3, mode)
    new = TokenLeadRefiner(16, 7, 2, 2, 1, 3, mode)
    new.load_state_dict(old.state_dict())
    expected = old(history, mask, base, aligned, available, estimates)
    actual = new(history[..., None], mask, base[..., None], aligned[..., None], available, estimates)
    torch.testing.assert_close(actual.squeeze(-1), expected)


def token_batch(length=48, horizon=12):
    generator = torch.Generator().manual_seed(81)
    history = torch.randn(1, length, 3, 3, generator=generator)*.15+.5
    future = torch.randn(1, horizon, 2, 2, generator=generator)*.15+.5
    htime = (torch.arange(length)[:, None, None]+torch.tensor([.1, .4, .8])[None, :, None])/(length+horizon)
    ftime = (length+torch.arange(horizon)[:, None, None]+torch.tensor([.1, .8])[None, :, None])/(length+horizon)
    return {'observed_data': history, 'observed_tp': htime[None].expand_as(history).clone(),
        'observed_mask': torch.ones_like(history, dtype=torch.bool),
        'future_exog': future, 'future_exog_tp': ftime[None].expand_as(future).clone(),
        'future_exog_mask': torch.ones_like(future, dtype=torch.bool),
        'query_patch_indices': torch.arange(horizon)[None],
        'tp_to_predict': (length+torch.arange(horizon)[None]+.4)/(length+horizon),
        'data_to_predict': torch.zeros(1, horizon, 1),
        'mask_predicted_data': torch.ones(1, horizon, 1, dtype=torch.bool)}


def config(variant='dag-lift'):
    return ModelConfig(1, 2, 48, variant=variant, future_patches=12, patch_encoder='ttcn',
        d_model=8, te_dim=4, n_heads=2, dropout=0, min_pairs=16, max_lag=5,
        min_correlation=0, leader_num=1, state_num=2)


@pytest.mark.parametrize('variant', ['dag-history', 'dag', 'dag-lift'])
def test_vector_model_no_scalar_means_hidden_nan_or_future_y_leakage_and_finite_gradients(variant, monkeypatch):
    import exobase.model
    def forbidden(*args):
        raise AssertionError('Scalar patch mean entered vector model')
    monkeypatch.setattr(exobase.model, 'scalar_patch_means', forbidden)
    torch.manual_seed(12)
    model = build_model(config(variant)).train()
    batch = token_batch()
    batch['observed_mask'][:, 5:7] = False
    batch['future_exog_mask'][:, 3:5] = False
    before, aux = model(batch)
    batch['observed_data'][~batch['observed_mask']] = float('nan')
    batch['observed_tp'][~batch['observed_mask']] = float('nan')
    batch['future_exog'][~batch['future_exog_mask']] = float('nan')
    batch['future_exog_tp'][~batch['future_exog_mask']] = float('nan')
    batch['data_to_predict'].fill_(float('nan'))
    batch['mask_predicted_data'].fill_(False)
    after, aux_after = model(batch)
    torch.testing.assert_close(after, before)
    torch.testing.assert_close(aux_after, aux)
    (before.square().mean()+aux).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.patch_encoder.Filter_Generators[0].weight.grad.abs().sum() > 0
    if variant == 'dag-lift':
        assert model.refiner.mix_layer.weight_real.grad.abs().sum() > 0


def test_end_to_end_prefix_prevents_future_control_effect_before_estimated_lag():
    torch.manual_seed(26)
    settings = config()
    settings.lag_estimator, settings.refiner_mode = 'pearson', 'prefix'
    model = build_model(settings).eval()
    batch = token_batch()
    estimates = model.estimate_lags(batch)
    assert estimates['leader_valid'].item()
    lag = estimates['leader_lag'].item()
    source = estimates['leaders'].item()
    before = model(batch)[0]
    batch['future_exog'][:, 3, :, source] += 2
    after = model(batch)[0]
    torch.testing.assert_close(before[:, :3+lag], after[:, :3+lag], atol=1e-6, rtol=0)
    assert (before[:, 3+lag:]-after[:, 3+lag:]).abs().max() > 1e-8
    # Lag export only needs historical fields, and is unaffected by future X.
    for key, value in model.estimate_lags(batch).items():
        torch.testing.assert_close(value, estimates[key])
    historical = {k: v for k, v in batch.items() if k.startswith('observed_')}
    for key, value in model.estimate_lags(historical).items():
        torch.testing.assert_close(value, estimates[key])


def test_no_valid_leader_falls_back_to_base_and_x_prediction_does_not_read_y():
    torch.manual_seed(16)
    model = build_model(config()).eval()
    batch = token_batch()
    batch['observed_mask'][..., 1:] = False
    history, mask = model.encode_history(batch)
    future, fm = model.patch_encoder(batch['future_exog'], batch['future_exog_tp'], batch['future_exog_mask'])
    base, px, _ = model.base_prediction(history, mask, future[..., :-1], fm)
    torch.testing.assert_close(model.forward_grid(batch)[0], base)
    history[:, :1] += 10
    _, px_after, _ = model.base_prediction(history, mask, future[..., :-1], fm)
    torch.testing.assert_close(px, px_after)


def test_constant_raw_control_cannot_become_a_leader_from_time_features():
    model = build_model(config()).eval()
    batch = token_batch()
    batch['observed_data'][..., 1:] = .5
    assert not model.estimate_lags(batch)['valid'].any()


@pytest.mark.parametrize('mode', ['prefix', 'global'])
@pytest.mark.parametrize('horizon,states', [(1, 1), (7, 3), (8, 1)])
def test_vector_refiner_time_axis_shapes_empty_fallback_and_gradients(mode, horizon, states):
    torch.manual_seed(72)
    module = TokenLeadRefiner(16, horizon, 2, 2, 3, states, mode)
    history = torch.randn(2, 2, 16, 3)
    mask = torch.ones(2, 2, 16, dtype=torch.bool)
    base = torch.randn(2, 2, horizon, 3, requires_grad=True)
    aligned = torch.randn(2, 2, 2, horizon, 3, requires_grad=True)
    available = torch.ones(2, 2, 2, horizon, dtype=torch.bool)
    estimates = {'leader_correlation': torch.full((2, 2, 2), .7),
                 'leader_valid': torch.ones(2, 2, 2, dtype=torch.bool)}
    result = module(history, mask, base, aligned, available, estimates)
    assert result.shape == base.shape
    result.square().mean().backward()
    assert torch.isfinite(base.grad).all()
    assert aligned.grad is None  # official LIFT detaches the aligned leader stream
    torch.testing.assert_close(module(history, mask, base, aligned,
        torch.zeros_like(available), estimates), base)


def test_queries_in_same_patch_can_decode_different_values():
    torch.manual_seed(6)
    model = build_model(config('dag-history')).eval()
    with torch.no_grad():
        model.patch_encoder.te_scale.weight.fill_(1)
        model.patch_encoder.te_scale.bias.zero_()
        for layer in (model.decoder.net[0], model.decoder.net[2], model.decoder.net[4]):
            layer.weight.zero_()
            layer.bias.zero_()
        model.decoder.net[0].weight[0, 8] = 1  # the linear actual-time embedding
        model.decoder.net[2].weight[0, 0] = 1
        model.decoder.net[4].weight[0, 0] = 1
    batch = token_batch()
    batch['query_patch_indices'] = torch.tensor([[0, 0]])
    batch['tp_to_predict'] = torch.tensor([[.8, .81]])
    result = model(batch)[0]
    assert (result[0, 1, 0]-result[0, 0, 0]).item() == pytest.approx(.01, abs=1e-6)
