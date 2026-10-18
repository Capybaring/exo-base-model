"""Numerical oracles from pinned upstream LIFT, plus missing-slot regressions."""
import ast
import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from exobase.lags import estimate_leads, ExogenousLeadRefiner
from exobase.lead_estimator import estimate_fft_leads, normalized_history, correlation_features
from exobase.model import ModelConfig, build_model
from exobase.token_lift import estimate_token_leads, align_token_exogenous, TokenLeadRefiner
from exobase.train import parser


def official_definitions():
    env = {'torch': torch, 'nn': nn, 'F': F, 'math': math}
    names = {'instance_norm', 'FilterFactory', 'ComplexLinear', 'LeadRefiner',
             'shifted_leader_seq', 'cross_corr_coef', 'estimate_indicator'}
    for path in ['upstream/lift/LIFT.py', 'upstream/lift/lead_estimate.py']:
        nodes = [n for n in ast.parse(Path(path).read_text()).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), env)
    return env


@pytest.mark.parametrize('length', [32, 33])
@pytest.mark.parametrize('scale', [1., 1e-6])
def test_scalar_fft_leaders_lags_and_signed_correlations_match_official(length, scale):
    torch.manual_seed(17)
    b, ny, nx, k = 2, 2, 3, 2
    x = torch.randn(b, nx, length)*scale
    y = torch.stack([x[:, 0].roll(3, -1), -x[:, 1].roll(5, -1)], 1)
    history = torch.cat([y, x], 1)
    mask = torch.ones_like(history, dtype=torch.bool)
    env = official_definitions()
    normalized = env['instance_norm'](history, -1)[0]
    ids, lags, r = env['estimate_indicator'](normalized, k,
        predefined_leaders=torch.arange(ny, ny+nx).expand(ny+nx, -1).contiguous())
    # These Pearson-only arguments intentionally cannot truncate/reject FFT.
    result = estimate_leads(y, x, mask[:, :ny], mask[:, ny:],
                            max_lag=1, min_pairs=10000, min_correlation=1, leader_num=k)
    torch.testing.assert_close(result['leaders'], ids[:, :ny]-ny)
    torch.testing.assert_close(result['leader_lag'], lags[:, :ny])
    torch.testing.assert_close(result['leader_correlation'], r[:, :ny], atol=2e-7, rtol=1e-6)
    assert (result['pair_count'] == length).all()
    assert result['leader_lag'][:, :, 0].tolist() == [[3, 5], [3, 5]]
    corr = env['cross_corr_coef'](normalized)[:, :ny, ny:]
    expected_shift = corr.abs().argmax(-1)
    torch.testing.assert_close(result['lag'], expected_shift+1)
    torch.testing.assert_close(result['correlation'], corr.gather(-1, expected_shift[..., None]).squeeze(-1))


def test_vector_fft_matches_direct_circular_dot_product_and_mask_pair_counts():
    torch.manual_seed(27)
    y, x = torch.randn(2, 2, 29, 4), torch.randn(2, 3, 29, 4)
    ym, xm = torch.rand(2, 2, 29) > .2, torch.rand(2, 3, 29) > .2
    yn, _ = normalized_history(y, ym)
    xn, _ = normalized_history(x, xm)
    direct = torch.stack([(yn[:, :, None]*xn[:, None].roll(ell, -2)).mean((-2, -1))
                          for ell in range(29)], -1)
    absolute = direct.abs()
    peaks = (absolute[..., 1:-1] >= absolute[..., :-2]) & (absolute[..., 1:-1] >= absolute[..., 2:])
    candidates = direct[..., 1:-1]*peaks
    lag = candidates.abs().argmax(-1)+1
    expected_corr = candidates.gather(-1, (lag-1)[..., None]).squeeze(-1)
    results = estimate_fft_leads(y, x, ym, xm, 2)
    torch.testing.assert_close(results['lag'], lag)
    torch.testing.assert_close(results['correlation'], expected_corr)
    counts = torch.stack([(ym[:, :, None] & xm[:, None].roll(ell, -1)).sum(-1)
                          for ell in range(29)], -1)
    torch.testing.assert_close(results['pair_count'], counts.gather(-1, lag[..., None]).squeeze(-1))
    hidden = estimate_fft_leads(y.masked_fill(~ym[..., None], float('nan')),
        x.masked_fill(~xm[..., None], float('nan')), ym, xm, 2)
    for key in results:
        torch.testing.assert_close(hidden[key], results[key])
    scalar = estimate_leads(y[..., 0], x[..., 0], ym, xm, leader_num=2)
    token = estimate_token_leads(y[..., :1], x[..., :1], ym, xm, 1, 999, 1, 2)
    for key in scalar:
        torch.testing.assert_close(token[key], scalar[key])


@pytest.mark.parametrize('states', [1, 3])
@pytest.mark.parametrize('temperature', [1., -.7, .03])
def test_multifeature_global_refiner_matches_official_per_feature(states, temperature):
    torch.manual_seed(38)
    b, ny, nx, length, horizon, p, k = 2, 2, 2, 16, 7, 3, 2
    history = torch.randn(b, ny+nx, length)
    tokens = history[..., None].expand(-1, -1, -1, p)
    future = torch.randn(b, ny+nx, horizon, p)
    mask = torch.ones(b, ny+nx, length, dtype=torch.bool)
    ids = torch.tensor([ny, ny+1]).expand(b, ny+nx, k).contiguous()
    lags = torch.tensor([3, 5]).expand_as(ids).contiguous()
    correlations = torch.tensor([.8, -.7]).expand_as(ids).contiguous()
    official = official_definitions()['LeadRefiner'](length, horizon, ny+nx, k, states)
    adapted = TokenLeadRefiner(length, horizon, ny, k, p, states, 'global')
    factory = official.factory.state_dict()
    if states > 1:
        factory['basic_state'] = factory['basic_state'][:ny]
        factory['classifier.weight'] = factory['classifier.weight'].unsqueeze(-1).expand(-1, -1, p).reshape(states, -1)/p
    adapted.factory.load_state_dict(factory)
    with torch.no_grad():
        official.temperature.fill_(temperature)
        adapted.temperature.fill_(temperature)
        adapted.mix_layer.weight_real.copy_(official.mix_layer.weight.real)
        adapted.mix_layer.weight_imag.copy_(official.mix_layer.weight.imag)
        adapted.mix_layer.bias_real.copy_(official.mix_layer.bias.real)
        adapted.mix_layer.bias_imag.copy_(official.mix_layer.bias.imag)
    leads = {'leaders': ids[:, :ny]-ny, 'leader_lag': lags[:, :ny],
             'leader_correlation': correlations[:, :ny],
             'leader_valid': torch.ones(b, ny, k, dtype=torch.bool)}
    aligned, available = align_token_exogenous(tokens[:, ny:], future[:, ny:], mask[:, ny:],
        torch.ones(b, nx, horizon, dtype=torch.bool), leads)
    expected = torch.stack([official(history, future[..., f], ids, lags, correlations)[:, :ny]
                            for f in range(p)], -1)
    actual = adapted(tokens[:, :ny], mask[:, :ny], future[:, :ny], aligned, available, leads)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


class InvalidDifferenceOnly(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.freq = horizon//2+1
        self.invalid_filter = nn.Parameter(torch.ones(()))

    def forward(self, history, feature):
        filters = history.new_zeros(*history.shape[:2], 5, self.freq)
        filters[..., 3, :] = self.invalid_filter
        return filters.flatten(-2), None


@pytest.mark.parametrize('mode', ['global', 'prefix'])
@pytest.mark.parametrize('vector', [False, True])
def test_invalid_or_completely_unavailable_leader_cannot_add_minus_base_stream(mode, vector):
    p, horizon = (3 if vector else 1), 5
    module = (TokenLeadRefiner(8, horizon, 1, 2, p, 1, mode) if vector
              else ExogenousLeadRefiner(8, horizon, 1, 2, 1, mode))
    module.factory = InvalidDifferenceOnly(horizon)
    with torch.no_grad():
        for parameter in module.mix_layer.parameters():
            parameter.zero_()
        freq = horizon//2+1
        module.mix_layer.weight_real[:, freq:2*freq].copy_(torch.eye(freq))
    history = torch.arange(8.).reshape(1, 1, 8)
    mask = torch.ones_like(history, dtype=torch.bool)
    base = torch.arange(1., 6.).reshape(1, 1, horizon)
    aligned = torch.zeros(1, 1, 2, horizon)
    available = torch.ones_like(aligned, dtype=torch.bool)
    leads = {'leader_correlation': torch.tensor([[[.8, 0.]]]),
             'leader_valid': torch.tensor([[[True, False]]])}
    if vector:
        history = history[..., None].expand(-1, -1, -1, p)
        base = base[..., None].expand(-1, -1, -1, p)
        aligned = aligned[..., None].expand(-1, -1, -1, -1, p)
    # Even a caller supplying True availability cannot resurrect an invalid slot.
    actual = module(history, mask, base, aligned, available, leads)
    torch.testing.assert_close(actual, base)
    actual.sum().backward()
    assert module.factory.invalid_filter.grad == 0
    feature = correlation_features(leads, module.temperature)
    assert feature[..., 1] == 0
    assert feature[..., 0].item() == pytest.approx(torch.tensor([1., .8]).softmax(0)[1].item())
    # A valid but entirely missing source must also be excluded from difference.
    leads['leader_valid'].fill_(True)
    available[..., 1, :] = False
    torch.testing.assert_close(module(history, mask, base, aligned, available, leads), base)
    available.fill_(False)
    torch.testing.assert_close(module(history, mask, base, aligned, available, leads), base)


def test_empty_constant_and_short_histories_are_handled_explicitly():
    torch.manual_seed(47)
    y, x = torch.randn(1, 1, 8, 3), torch.randn(1, 2, 8, 3)
    ym, xm = torch.ones(1, 1, 8, dtype=torch.bool), torch.ones(1, 2, 8, dtype=torch.bool)
    xm[:, 0] = False
    x[:, 1] = 4
    leads = estimate_fft_leads(y, x, ym, xm, 2)
    assert not leads['valid'].any() and not leads['lag'].any()
    with pytest.raises(ValueError, match='at least 3'):
        estimate_fft_leads(y[..., :2, :], x[..., :2, :], ym[..., :2], xm[..., :2], 2)


def test_defaults_presets_and_saved_checkpoint_estimator_selection():
    defaults = parser().parse_args([])
    assert defaults.lag_estimator == 'fft' and defaults.refiner_mode == 'global'
    config = ModelConfig(1, 1, 8, future_patches=4, d_model=8, n_heads=2, patch_encoder='ttcn')
    assert config.lag_estimator == 'fft' and config.refiner_mode == 'global'
    # No default min_pairs=8 rejection for an otherwise valid eight-patch FFT input.
    build_model(config)
    fresh = asdict(config)
    fresh.pop('lag_estimator')
    assert build_model(fresh).config.lag_estimator == 'fft'
    for path in Path('configs').rglob('*.json'):
        preset = json.loads(path.read_text())
        if 'refiner_mode' in preset:
            assert preset['lag_estimator'] == 'fft' and preset['refiner_mode'] == 'global'
            assert not {'min_pairs', 'min_correlation', 'max_lag'} & preset.keys()
    saved = asdict(config)
    assert ModelConfig.from_saved(saved).lag_estimator == 'fft'
    saved.pop('lag_estimator')
    saved.update(min_pairs=3, refiner_mode='prefix')
    reloaded = ModelConfig.from_saved(saved)
    assert reloaded.lag_estimator == 'pearson' and reloaded.refiner_mode == 'prefix'
    old = build_model(reloaded)
    old.load_state_dict(build_model(ModelConfig.from_saved(saved)).state_dict())
