"""Shared config/factory and the explicitly retained legacy t-PatchGNN models.

New default training routes to y_forecast.py. DAG and legacy classes preserve
old checkpoints and comparisons; their graphs are NOT used by v4 variants.
"""
import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from .upstream_model import tPatchGNN


@dataclass
class ModelConfig:
    target_dim: int
    exog_dim: int
    history_patches: int
    variant: str = 'dag-lift'
    d_model: int = 64
    te_dim: int = 10
    n_heads: int = 4
    n_layers: int = 1
    tf_layers: int = 1
    node_dim: int = 10
    graph_hops: int = 2
    delay_context: int = 4
    future_patches: int = 0
    alpha: float = .5
    beta: float = .1
    dropout: float = .1
    max_lag: int = 48
    min_pairs: int = 8
    min_correlation: float = .3
    leader_num: int = 3
    state_num: int = 8
    refiner_mode: str = 'global'
    lag_estimator: str = 'fft'
    # Missing from v2 checkpoint configs: retain their scalar-mean architecture.
    # New training explicitly selects ttcn in train.py.
    patch_encoder: str = 'mean'
    # Execution scheduling only; absent from older checkpoints -> old behavior.
    ttcn_execution: str = 'chunked'

    @classmethod
    def from_saved(cls, settings):
        """Older checkpoints predate FFT estimation; preserve their selection."""
        settings = dict(settings)
        settings.setdefault('lag_estimator', 'pearson')
        settings.setdefault('refiner_mode', 'prefix')
        return cls(**settings)


class TPatchBackbone(tPatchGNN):
    """Portability-safe TTCN; otherwise the upstream model is unchanged."""
    def TTCN(self, features, mask):
        valid = mask.bool()
        features = torch.where(valid, features, torch.zeros_like(features))
        logits = self.Filter_Generators(features)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = logits.softmax(dim=1) * valid.to(logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights = weights.reshape(*features.shape[:2], self.ttcn_dim, -1)
        result = (features.unsqueeze(-2)*weights).sum((1, 3))
        return torch.relu(result+self.T_bias)*valid.any(dim=1)

    def _features(self, values, times, mask):
        # [B,K,L,C] -> [B*C*K,L,1]. Real times stay in the shared record clock.
        b, k, length, c = values.shape
        v = values.permute(0, 3, 1, 2).reshape(-1, length, 1)
        t = times.permute(0, 3, 1, 2).reshape(-1, length, 1)
        m = mask.permute(0, 3, 1, 2).reshape(-1, length, 1).bool()
        v = torch.where(m, v, torch.zeros_like(v))
        t = torch.where(m, t, torch.zeros_like(t))
        if not torch.isfinite(v).all() or not torch.isfinite(t).all():
            raise ValueError('Observed values/times must be finite')
        return torch.cat([v, self.LearnableTE(t)], dim=-1), m

    def encode_history(self, values, times, mask):
        self.batch_size = values.shape[0]
        features, m = self._features(values, times, mask)
        return self.IMTS_Model(features, m)

    def encode_patch_tokens(self, values, times, mask):
        b, k, _, c = values.shape
        features, m = self._features(values, times, mask)
        valid = m.any(dim=1)
        h = torch.cat([self.TTCN(features, m), valid.to(features.dtype)], dim=-1)
        return h.reshape(b, c, k, -1), valid.reshape(b, c, k)


def scalar_patch_means(values, mask):
    """Observed means/masks per fixed-duration bin, returned as [B,C,K]."""
    valid = mask.bool()
    safe = torch.where(valid, values, torch.zeros_like(values))
    count = valid.sum(dim=2)
    return (safe.sum(dim=2)/count.clamp_min(1)).transpose(1, 2), (count > 0).transpose(1, 2)


def pattern_windows(batch, target_dim, context):
    past, pm = scalar_patch_means(batch['observed_data'][..., target_dim:],
                                  batch['observed_mask'][..., target_dim:])
    future, fm = scalar_patch_means(batch['future_exog'], batch['future_exog_mask'])
    x, valid = torch.cat([past, future], 2), torch.cat([pm, fm], 2)
    x = torch.nn.functional.pad(x, (context-1, 0))
    valid = torch.nn.functional.pad(valid, (context-1, 0), value=False)
    windows = x.unfold(2, context, 1)
    # A pattern retrieval window must have one observed summary in every bin.
    complete = valid.unfold(2, context, 1).all(dim=-1)
    return windows, complete


def z_normalize(x):
    return (x-x.mean(dim=-1, keepdim=True))/x.std(dim=-1, correction=0, keepdim=True).clamp_min(1e-8)


class PatternMemory(nn.Module):
    """PDFormer pattern attention; its result changes only the exogenous Key.

    The explicit IMTS adaptation is k-Shape on scalar means over fixed-duration
    patches, rather than raw regularly sampled traffic values. Main-path TTCN
    inputs are never interpolated or resampled.
    """
    def __init__(self, patterns, context, dim):
        super().__init__()
        patterns = torch.as_tensor(patterns, dtype=torch.float32)
        if patterns.ndim != 2 or patterns.shape[1] != context or not len(patterns):
            raise ValueError('Pattern bank must be nonempty [patterns, context]')
        self.register_buffer('patterns', patterns)
        self.embedding = nn.Linear(context, dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim**-.5

    def forward(self, windows, valid):
        q = self.query(self.embedding(z_normalize(windows)))
        memory = self.embedding(z_normalize(self.patterns))
        weights = (q @ self.key(memory).T * self.scale).softmax(dim=-1)
        return (weights @ self.value(memory))*valid.unsqueeze(-1)


def gather_query_patch(tokens, indices):
    # [B,N,K,D], [B,Q] -> [B,Q,N,D]
    b, n, _, d = tokens.shape
    return tokens.gather(2, indices[:, None, :, None].expand(b, n, -1, d)).transpose(1, 2)


class ExogenousTPatchGNN(nn.Module):
    def __init__(self, config, patterns=None):
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig(**config)
        if config.variant not in ('tpatch-y', 'tpatch', 'exo', 'delay'):
            raise ValueError('Unknown model variant')
        if (config.d_model < 2 or config.d_model % 2 or config.te_dim < 2
                or config.n_heads < 1 or config.d_model % config.n_heads):
            raise ValueError('d_model must be even, >=2 and divisible by n_heads; te_dim >=2')
        if min(config.target_dim, config.exog_dim, config.history_patches,
               config.n_layers, config.tf_layers, config.graph_hops, config.delay_context) < 1:
            raise ValueError('Model dimensions and layer counts must be positive')
        self.config = config
        n = config.target_dim if config.variant == 'tpatch-y' else config.target_dim+config.exog_dim
        args = SimpleNamespace(device='cpu', hid_dim=config.d_model, ndim=n,
            npatch=config.history_patches, nlayer=config.n_layers, te_dim=config.te_dim,
            nhead=config.n_heads, tf_layer=config.tf_layers, node_dim=config.node_dim,
            hop=config.graph_hops, outlayer='Linear')
        self.backbone = TPatchBackbone(args)
        if config.variant in ('exo', 'delay'):
            d = config.d_model
            self.exog_embedding = nn.Embedding(config.exog_dim, d)
            self.history_projection = nn.Linear(d, d, bias=False)
            self.time_projection = nn.Linear(config.te_dim, d)
            self.query = nn.Linear(d, d)
            self.key = nn.Linear(d, d)
            self.value = nn.Linear(d, d)
        if config.variant == 'delay':
            self.memory = PatternMemory(patterns, config.delay_context, config.d_model)

    def forward(self, batch):
        c = self.config
        # Targets and target masks are intentionally never read in forward().
        v, t, m = (batch[k] for k in ('observed_data', 'observed_tp', 'observed_mask'))
        if c.variant == 'tpatch-y':
            v, t, m = v[..., :c.target_dim], t[..., :c.target_dim], m[..., :c.target_dim]
        state = self.backbone.encode_history(v, t, m)
        te = self.backbone.LearnableTE(batch['tp_to_predict'].unsqueeze(-1))
        y = state[:, :c.target_dim].unsqueeze(2).expand(-1, -1, te.shape[1], -1)
        if c.variant in ('exo', 'delay'):
            tokens, valid = self.backbone.encode_patch_tokens(
                batch['future_exog'], batch['future_exog_tp'], batch['future_exog_mask'])
            ix = batch['query_patch_indices']
            at_query = gather_query_patch(tokens, ix)
            at_query = at_query+self.history_projection(state[:, c.target_dim:])[:, None]
            at_query = at_query+self.exog_embedding.weight[None, None]
            key, value = self.key(at_query), self.value(at_query)
            if c.variant == 'delay':
                windows, complete = pattern_windows(batch, c.target_dim, c.delay_context)
                retrieved = self.memory(windows, complete)
                future_retrieved = retrieved[:, :, v.shape[1]:]
                key = key+gather_query_patch(future_retrieved, ix)
            q = self.query(y.transpose(1, 2)+self.time_projection(te)[:, :, None])
            scores = torch.einsum('bqjd,bqid->bqji', q, key)/math.sqrt(c.d_model)
            available = valid.gather(2, ix[:, None].expand(-1, valid.shape[1], -1)).transpose(1, 2)
            available = available[:, :, None]
            scores = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
            alpha = scores.softmax(-1)*available
            alpha = alpha/alpha.sum(-1, keepdim=True).clamp_min(1e-12)
            context = torch.einsum('bqji,bqid->bqjd', alpha, value)
            y = y+context.transpose(1, 2)
        time = te[:, None].expand(-1, c.target_dim, -1, -1)
        prediction = self.backbone.decoder(torch.cat([y, time], -1)).squeeze(-1).transpose(1, 2)
        return prediction.contiguous(), prediction.new_zeros(())


def build_model(config, patterns=None):
    """Route Y/patch/LIFT models, retained DAG, and legacy baselines."""
    if isinstance(config, dict):
        config = ModelConfig(**config)
    if config.variant in ('y-base', 'patch-base', 'patch-lift'):
        from .y_forecast import YForecast
        return YForecast(config)
    if config.variant in ('dag-history', 'dag', 'dag-lift'):
        if config.patch_encoder == 'ttcn':
            from .token_dag import DAGTokenForecast
            return DAGTokenForecast(config)
        if config.patch_encoder != 'mean':
            raise ValueError('Unknown patch encoder')
        from .dag_model import DAGExogenousForecast
        return DAGExogenousForecast(config)
    return ExogenousTPatchGNN(config, patterns)
