"""TTCN vector patches + DAG dual attention + directed token LIFT.

This is a vector adaptation of DAG, not unchanged scalar DAG. Temporal tokens
preserve every TTCN feature. Covariate tokens encode whole feature sequences.
Known future X reaches dag-lift outputs only through aligned LIFT.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .ttcn import TTCNPatchEncoder, QueryDecoder
from .token_lift import token_statistics, estimate_token_leads, align_token_exogenous, TokenLeadRefiner
from .lead_estimator import validate_lead_settings
from .dag_layers.Embed import CompressAndProject, PositionalEmbedding
from .dag_layers.SelfAttention_Family import FullAttention, AttentionLayer
from .dag_layers.Transformer_EncDec import Encoder, EncoderLayer


def encoder(config, output_attention):
    c = config
    return Encoder([EncoderLayer(AttentionLayer(FullAttention(False,
        attention_dropout=c.dropout, output_attention=output_attention), c.d_model, c.n_heads),
        c.d_model, 4*c.d_model, c.dropout, 'gelu') for _ in range(c.n_layers)],
        norm_layer=nn.LayerNorm(c.d_model))


def normalized_tokens(features, mask, mean, std):
    features = torch.where(mask[..., None], (features-mean)/std, 0)
    return torch.cat([features, mask[..., None].to(features.dtype)], -1)


class VectorEmbedding(nn.Module):
    def __init__(self, steps, channels, dim, dropout):
        super().__init__()
        self.value = nn.Linear(steps*dim, dim)
        self.context = CompressAndProject(channels*dim, steps, dim//4)
        self.project = nn.Linear(dim+dim//4, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        b, c, t, d = tokens.shape
        context = self.context(tokens.permute(0, 2, 1, 3).reshape(b, t, c*d))
        context = context[:, None].expand(-1, c, -1)
        return self.dropout(self.project(torch.cat([self.value(tokens.flatten(-2)), context], -1)))


class VectorProjection(nn.Module):
    def __init__(self, channels, dim, steps):
        super().__init__()
        self.features, self.steps = dim-1, steps
        self.variables = nn.Linear(channels, 1)
        self.sequence = nn.Linear(dim, steps*(dim-1))

    def forward(self, tokens):
        reduced = self.variables(tokens.transpose(1, 2)).squeeze(-1)
        return self.sequence(reduced).reshape(-1, self.steps, self.features)


class DAGTokenForecast(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = self.config = config
        if (c.variant not in ('dag-history', 'dag', 'dag-lift') or c.patch_encoder != 'ttcn'
                or c.d_model < 4 or c.d_model % 2 or c.n_heads < 1
                or c.d_model % c.n_heads or c.te_dim < 2):
            raise ValueError('Invalid TTCN/DAG variant or embedding dimensions')
        if min(c.target_dim, c.exog_dim, c.history_patches, c.future_patches,
               c.n_layers, c.leader_num, c.state_num) < 1:
            raise ValueError('Model dimensions must be positive')
        validate_lead_settings(c)
        if not 0 <= c.alpha <= 1 or c.beta < 0 or not 0 <= c.dropout < 1:
            raise ValueError('Invalid mixture, auxiliary weight or dropout')
        d, p, length, horizon = c.d_model, c.d_model-1, c.history_patches, c.future_patches
        self.patch_encoder = TTCNPatchEncoder(d, c.te_dim)
        self.decoder = QueryDecoder(d, c.te_dim)
        self.position = PositionalEmbedding(d)
        self.temporal_x = encoder(c, True)
        self.temporal_y = encoder(c, False)
        self.head_x = nn.Linear(length*d, horizon*p)
        self.head_y = nn.Linear(length*d, horizon*p)
        self.temporal_context_x = CompressAndProject(c.exog_dim*d, length, d)
        self.temporal_context_y = CompressAndProject(c.target_dim*d, length, d)
        self.cov_history_embedding = VectorEmbedding(length, c.exog_dim, d, c.dropout)
        self.cov_future_embedding = VectorEmbedding(horizon, c.exog_dim, d, c.dropout)
        self.cov_history = encoder(c, True)
        self.cov_future = encoder(c, False)
        self.cov_context_history = CompressAndProject(c.exog_dim*d, length, d)
        self.cov_context_future = CompressAndProject(c.exog_dim*d, horizon, d)
        self.reconstruct = VectorProjection(c.exog_dim, d, length)
        self.predict_covariate = VectorProjection(c.exog_dim, d, horizon)
        if c.variant == 'dag-lift':
            self.refiner = TokenLeadRefiner(length, horizon, c.target_dim,
                min(c.leader_num, c.exog_dim), p, c.state_num, c.refiner_mode)

    def encode_history(self, batch):
        tokens, mask = self.patch_encoder(batch['observed_data'], batch['observed_tp'], batch['observed_mask'])
        c = self.config
        if tokens.shape[1:3] != (c.target_dim+c.exog_dim, c.history_patches):
            raise ValueError('History patch grid differs from saved configuration')
        return tokens[..., :-1], mask

    @torch.no_grad()
    def estimate_from_tokens(self, history, mask, batch):
        c = self.config
        # Absolute time features can vary even when a physical variable is
        # constant. Such a channel is not an eligible predictive leader/target.
        raw_mask = batch['observed_mask'].bool()
        raw = torch.where(raw_mask, batch['observed_data'], 0)
        count = raw_mask.sum((1, 2)).clamp_min(1)
        center = raw.sum((1, 2))/count
        variation = torch.where(raw_mask, raw-center[:, None, None], 0).square().sum((1, 2))
        mask = mask & (variation > 1e-10)[..., None]
        return estimate_token_leads(history[:, :c.target_dim], history[:, c.target_dim:],
            mask[:, :c.target_dim], mask[:, c.target_dim:], c.max_lag, c.min_pairs,
            c.min_correlation, c.leader_num, estimator=c.lag_estimator)

    def estimate_lags(self, batch):
        history, mask = self.encode_history(batch)
        return self.estimate_from_tokens(history, mask, batch)

    @staticmethod
    def context_tokens(tokens):
        b, c, t, d = tokens.shape
        return tokens.permute(0, 2, 1, 3).reshape(b, t, c*d)

    def base_prediction(self, history, mask, future, fm):
        c = self.config
        b, _, length, p = history.shape
        yh, xh = history[:, :c.target_dim], history[:, c.target_dim:]
        ym, xm = mask[:, :c.target_dim], mask[:, c.target_dim:]
        _, my, sy = token_statistics(yh, ym)
        _, mx, sx = token_statistics(xh, xm)
        my, sy, mx, sx = [v.detach() for v in (my, sy, mx, sx)]
        yt, xt = normalized_tokens(yh, ym, my, sy), normalized_tokens(xh, xm, mx, sx)
        y_flat, x_flat = yt.reshape(-1, length, c.d_model), xt.reshape(-1, length, c.d_model)
        y_flat = y_flat+self.position(y_flat)
        x_flat = x_flat+self.position(x_flat)
        exog_encoded, _ = self.temporal_x(x_flat)
        _, transfer = self.temporal_x(y_flat)
        gate = torch.sigmoid((self.temporal_context_y(self.context_tokens(yt))*
                              self.temporal_context_x(self.context_tokens(xt))).sum(-1))
        gate = gate.repeat_interleave(c.target_dim).view(-1, 1, 1, 1)
        endogenous, _ = self.temporal_y(y_flat, exog_attns=transfer, attn_alpha=gate)
        px = self.head_x(exog_encoded.flatten(1)).reshape(b, c.exog_dim, c.future_patches, p)*sx+mx
        py = self.head_y(endogenous.flatten(1)).reshape(b, c.target_dim, c.future_patches, p)*sy+my
        if c.variant == 'dag':
            supplied, supplied_mask = future, fm
        else:
            supplied = px
            supplied_mask = torch.ones(px.shape[:-1], dtype=torch.bool, device=px.device)
        x_normalized = torch.where(xm[..., None], (xh-mx)/sx, 0)
        f_normalized = torch.where(supplied_mask[..., None], (supplied-mx)/sx, 0)
        covariate, reconstructed = [], []
        for j in range(c.target_dim):
            past_features = torch.where(xm[..., None], x_normalized*sy[:, j:j+1]+my[:, j:j+1], 0)
            next_features = torch.where(supplied_mask[..., None], f_normalized*sy[:, j:j+1]+my[:, j:j+1], 0)
            past = torch.cat([past_features, xm[..., None].to(history.dtype)], -1)
            next_ = torch.cat([next_features, supplied_mask[..., None].to(history.dtype)], -1)
            past_embedding = self.cov_history_embedding(past)
            next_embedding = self.cov_future_embedding(next_)
            past_state, _ = self.cov_history(past_embedding)
            _, transfer = self.cov_history(next_embedding)
            gate = torch.sigmoid((self.cov_context_history(self.context_tokens(past))*
                                  self.cov_context_future(self.context_tokens(next_))).sum(-1))
            next_state, _ = self.cov_future(next_embedding, exog_attns=transfer,
                                           attn_alpha=gate.view(b, 1, 1, 1))
            reconstructed.append(self.reconstruct(past_state))
            covariate.append(self.predict_covariate(next_state))
        covariate = torch.stack(covariate, 1)
        reconstructed = torch.stack(reconstructed, 1)
        return c.alpha*py+(1-c.alpha)*covariate, px, reconstructed

    def point_error(self, grid, indices, times, truth):
        features = grid[indices[:, 0], indices[:, 3], indices[:, 1]]
        prediction = self.decoder(features, self.patch_encoder.time_embedding(times[:, None]))
        return (prediction-truth).square().sum()

    def observed_point_loss(self, grid, values, times, mask):
        # Decode only observed points. Recompute decoder chunks during backward
        # instead of materializing [B,K,all_points,C,D] auxiliary activations.
        indices = mask.nonzero()
        total = grid.new_zeros(())
        for start in range(0, len(indices), 8192):
            selected = indices[start:start+8192]
            at = tuple(selected.unbind(-1))
            total = total+checkpoint(self.point_error, grid, selected, times[at], values[at],
                                     use_reentrant=False)
        return total/max(len(indices), 1)

    def auxiliary_loss(self, batch, predicted_x, reconstructed_y):
        c = self.config
        xm = batch['future_exog_mask'].bool()
        ym = batch['observed_mask'][..., :c.target_dim].bool()
        return c.beta*(self.observed_point_loss(predicted_x, batch['future_exog'],
            batch['future_exog_tp'], xm)+self.observed_point_loss(reconstructed_y,
            batch['observed_data'][..., :c.target_dim], batch['observed_tp'][..., :c.target_dim], ym))

    def forward_grid(self, batch):
        c = self.config
        history, mask = self.encode_history(batch)
        future, fm = self.patch_encoder(batch['future_exog'], batch['future_exog_tp'], batch['future_exog_mask'])
        future = future[..., :-1]
        if future.shape[1:3] != (c.exog_dim, c.future_patches):
            raise ValueError('Future patch grid differs from saved configuration')
        base, predicted_x, reconstructed_y = self.base_prediction(history, mask, future, fm)
        if c.variant == 'dag-lift':
            leads = self.estimate_from_tokens(history, mask, batch)
            aligned, available = align_token_exogenous(history[:, c.target_dim:], future,
                mask[:, c.target_dim:], fm, leads)
            base = self.refiner(history[:, :c.target_dim], mask[:, :c.target_dim],
                                base, aligned, available, leads)
        auxiliary = (self.auxiliary_loss(batch, predicted_x, reconstructed_y)
                     if self.training and c.beta else base.new_zeros(()))
        return base, auxiliary

    def decode_queries(self, grid, indices, query_times):
        b, channels, _, p = grid.shape
        at_query = grid.gather(2, indices[:, None, :, None].expand(b, channels, -1, p)).transpose(1, 2)
        times = self.patch_encoder.time_embedding(query_times[..., None])
        times = times[:, :, None].expand(-1, -1, channels, -1)
        return self.decoder(at_query, times)

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
