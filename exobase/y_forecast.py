"""Y-only temporal baseline, unchanged TTCN patches, and optional X->Y LIFT.

No X forecasting, Y reconstruction, DAG transfer, or covariate backbone.
Lags use scalar observed patch summaries rather than independently learned
X/Y latent coordinates. Main TTCN paths retain their full feature vectors.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .model import scalar_patch_means
from .ttcn import TTCNPatchEncoder, QueryDecoder
from .dag_layers.Embed import PositionalEmbedding
from .lead_estimator import validate_lead_settings
from .token_lift import (token_statistics, estimate_token_leads,
                         align_token_exogenous, TokenLeadRefiner)


VARIANTS = ('y-base', 'patch-base', 'patch-lift')


class MeanPatchEncoder(nn.Module):
    """Fixed-bin scalar mean baseline; no learnable point pooling/TTCN."""
    def __init__(self, dim, te_dim):
        super().__init__()
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, te_dim-1)
        self.projection = nn.Linear(1+te_dim, dim-1)

    def time_embedding(self, time):
        return torch.cat([self.te_scale(time), self.te_periodic(time).sin()], -1)

    def forward(self, values, times, mask):
        means, present = scalar_patch_means(values, mask)
        mean_times, _ = scalar_patch_means(times, mask)
        hidden = self.projection(torch.cat([means[..., None],
            self.time_embedding(mean_times[..., None])], -1)).relu()*present[..., None]
        return torch.cat([hidden, present[..., None].to(hidden.dtype)], -1), present


class YForecast(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = self.config = config
        if c.variant not in VARIANTS:
            raise ValueError('Unknown Y temporal variant')
        if (c.d_model < 4 or c.d_model % 2 or c.n_heads < 1
                or c.d_model % c.n_heads or c.te_dim < 2
                or min(c.target_dim, c.history_patches, c.future_patches, c.n_layers) < 1
                or not 0 <= c.dropout < 1):
            raise ValueError('Invalid Y temporal model dimensions or dropout')
        expected = 'mean' if c.variant == 'y-base' else 'ttcn'
        if c.patch_encoder != expected:
            raise ValueError(f'{c.variant} requires patch_encoder={expected}')
        d, p = c.d_model, c.d_model-1
        encoder = MeanPatchEncoder if c.variant == 'y-base' else TTCNPatchEncoder
        self.y_encoder = encoder(d, c.te_dim)
        self.position = PositionalEmbedding(d, max_len=max(5000, c.history_patches))
        layer = nn.TransformerEncoderLayer(d, c.n_heads, 4*d, c.dropout,
            activation='gelu', batch_first=True)
        self.temporal = nn.TransformerEncoder(layer, c.n_layers,
            norm=nn.LayerNorm(d), enable_nested_tensor=False)
        # A compact direct multi-step head, shared across Y variables.
        self.time_projection = nn.Linear(c.history_patches, c.future_patches)
        self.feature_projection = nn.Linear(d, p)
        self.decoder = QueryDecoder(d, c.te_dim)
        # Initialize the whole Y backbone before optional X modules so matched
        # seeds give patch-base and patch-lift identical initial Y parameters.
        if c.variant == 'patch-lift':
            if min(c.exog_dim, c.leader_num, c.state_num) < 1:
                raise ValueError('LIFT dimensions must be positive')
            validate_lead_settings(c)
            self.x_encoder = TTCNPatchEncoder(d, c.te_dim)
            self.x_to_y = nn.Linear(p, p, bias=False)
            self.refiner = TokenLeadRefiner(c.history_patches, c.future_patches,
                c.target_dim, min(c.leader_num, c.exog_dim), p, c.state_num,
                c.refiner_mode, detach_leaders=False)

    def encode_y(self, batch):
        c = self.config
        tokens, mask = self.y_encoder(*(batch[k][..., :c.target_dim]
            for k in ('observed_data', 'observed_tp', 'observed_mask')))
        if tokens.shape[1:3] != (c.target_dim, c.history_patches):
            raise ValueError('History patch grid differs from saved configuration')
        return tokens[..., :-1], mask

    def base_prediction(self, history, mask):
        c = self.config
        normalized, mean, std = token_statistics(history, mask)
        mean, std = mean.detach(), std.detach()
        normalized = torch.where(mask[..., None], (normalized-mean)/std, 0)
        tokens = torch.cat([normalized, mask[..., None].to(history.dtype)], -1)
        flat = tokens.flatten(0, 1)
        missing = ~mask.flatten(0, 1)
        # All-empty histories still have a finite, learned default prediction.
        missing = missing.clone()
        missing[missing.all(-1), 0] = False
        state = self.temporal(flat+self.position(flat), src_key_padding_mask=missing)
        future = self.time_projection(state.transpose(1, 2)).transpose(1, 2)
        future = self.feature_projection(future).reshape(history.shape[0],
            c.target_dim, c.future_patches, c.d_model-1)
        return future*std+mean

    @torch.no_grad()
    def estimate_lags(self, batch):
        if self.config.variant != 'patch-lift':
            raise ValueError('Only patch-lift estimates X->Y lags')
        c = self.config
        means, mask = scalar_patch_means(batch['observed_data'], batch['observed_mask'])
        raw_mask = batch['observed_mask'].bool()
        raw = torch.where(raw_mask, batch['observed_data'], 0)
        count = raw_mask.sum((1, 2)).clamp_min(1)
        center = raw.sum((1, 2))/count
        variation = torch.where(raw_mask, raw-center[:, None, None], 0).square().sum((1, 2))
        # Uneven point counts can produce tiny floating-point differences in
        # otherwise constant patch means; they are not predictive signals.
        mask = mask & (variation > 1e-10)[..., None]
        # The lag diagnostic is independent of neural features and timestamps.
        # Empty bins are kept, not compressed or interpolated.
        return estimate_token_leads(means[:, :c.target_dim, :, None],
            means[:, c.target_dim:, :, None], mask[:, :c.target_dim],
            mask[:, c.target_dim:], c.max_lag, c.min_pairs, c.min_correlation,
            c.leader_num, estimator=c.lag_estimator)

    def forward_grid(self, batch):
        c = self.config
        history, mask = self.encode_y(batch)
        base = self.base_prediction(history, mask)
        if c.variant == 'patch-lift':
            xh, xm = self.x_encoder(*(batch[k][..., c.target_dim:]
                for k in ('observed_data', 'observed_tp', 'observed_mask')))
            xf, fm = self.x_encoder(*(batch[k]
                for k in ('future_exog', 'future_exog_tp', 'future_exog_mask')))
            if xf.shape[1:3] != (c.exog_dim, c.future_patches):
                raise ValueError('Future X grid differs from saved configuration')
            leads = self.estimate_lags(batch)
            aligned, available = align_token_exogenous(xh[..., :-1], xf[..., :-1], xm, fm, leads)
            # Learned map into the Y correction coordinates, after per-X
            # history normalization and signed lead alignment. No detach here.
            aligned = self.x_to_y(aligned)
            base = self.refiner(history, mask, base, aligned, available, leads)
        return base, base.new_zeros(())

    def decode_queries(self, grid, indices, times):
        b, channels, _, p = grid.shape
        features = grid.gather(2, indices[:, None, :, None].expand(b, channels, -1, p)).transpose(1, 2)
        te = self.y_encoder.time_embedding(times[..., None])
        te = te[:, :, None].expand(-1, -1, channels, -1)
        return self.decoder(features, te)

    def forward(self, batch):
        grid, auxiliary = self.forward_grid(batch)
        predictions = []
        for start in range(0, batch['query_patch_indices'].shape[1], 512):
            indices = batch['query_patch_indices'][:, start:start+512]
            times = batch['tp_to_predict'][:, start:start+512]
            if self.training and torch.is_grad_enabled():
                predictions.append(checkpoint(self.decode_queries, grid, indices, times, use_reentrant=False))
            else:
                predictions.append(self.decode_queries(grid, indices, times))
        return torch.cat(predictions, 1), auxiliary
