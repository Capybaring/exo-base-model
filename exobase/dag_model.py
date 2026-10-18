"""DAG dual attention on real-time patch summaries, with optional X->Y LIFT.

The irregular adapter uses t-PatchGNN's physical-time grouping, not its GNN.
Patch means/masks form the regular interface; no learned TTCN/GNN is used in
these variants. DAG's original temporal/covariate attention components remain.
"""
from types import SimpleNamespace

import torch
from torch import nn

from .dag_layers.dag_model import DAGModel
from .lags import estimate_leads, align_exogenous, ExogenousLeadRefiner
from .lead_estimator import validate_lead_settings
from .model import scalar_patch_means


class MaskedCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, prediction, target):
        error = torch.where(self.mask, prediction - target, 0)
        return error.square().sum() / self.mask.sum().clamp_min(1)


def fill_history_means(sequence, mask):
    """[B,L,C]: empty cells use that channel's observed history mean (else 0)."""
    safe = torch.where(mask, sequence, 0)
    mean = safe.sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
    return torch.where(mask, safe, mean), mean


class DAGExogenousForecast(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = self.config = config
        if c.variant not in ('dag-history', 'dag', 'dag-lift'):
            raise ValueError('Unknown DAG variant')
        if min(c.target_dim, c.exog_dim, c.history_patches, c.future_patches,
               c.n_layers, c.n_heads, c.d_model, c.leader_num, c.state_num) < 1:
            raise ValueError('Model sizes and lag settings must be positive')
        if c.d_model < 4 or c.d_model % 2 or c.d_model % c.n_heads:
            raise ValueError('DAG requires even d_model >=4, divisible by n_heads')
        validate_lead_settings(c)
        if not 0 <= c.alpha <= 1 or c.beta < 0 or not 0 <= c.dropout < 1:
            raise ValueError('Invalid DAG mixture, auxiliary weight or dropout')
        # Width-one DAG embedding does not regroup the real-time bins. The
        # upstream embedding retains its original final replicated boundary bin.
        args = SimpleNamespace(seq_len=c.history_patches, pred_len=c.future_patches,
            patch_len=1, stride=1, use_c=True, use_t=True, use_c_exog=True,
            use_t_exog=True, alpha=c.alpha, beta=c.beta, series_dim=c.target_dim,
            infer_use_future=True, enc_in=c.target_dim + c.exog_dim,
            d_model=c.d_model, d_ff=4*c.d_model, n_heads=c.n_heads,
            e_layers=c.n_layers, dropout=c.dropout, factor=5, activation='gelu',
            criterion=nn.MSELoss())
        self.core = DAGModel(args)
        self.core.temporal_encoder.criterion = MaskedCriterion()
        self.core.cov_encoder.criterion = MaskedCriterion()
        if c.variant == 'dag-lift':
            self.refiner = ExogenousLeadRefiner(c.history_patches, c.future_patches,
                c.target_dim, min(c.leader_num, c.exog_dim), c.state_num, c.refiner_mode)

    def regularize(self, batch):
        history, hm = scalar_patch_means(batch['observed_data'], batch['observed_mask'])
        future, fm = scalar_patch_means(batch['future_exog'], batch['future_exog_mask'])
        c = self.config
        if (history.shape[1:] != (c.target_dim + c.exog_dim, c.history_patches)
                or future.shape[1:] != (c.exog_dim, c.future_patches)):
            raise ValueError('Batch patch grid differs from the saved model configuration')
        if not torch.isfinite(history).all() or not torch.isfinite(future).all():
            raise ValueError('Observed patch summaries must be finite')
        return history, hm, future, fm

    def estimate_lags(self, batch):
        c = self.config
        # This public diagnostic API does not even require future-X fields.
        history, hm = scalar_patch_means(batch['observed_data'], batch['observed_mask'])
        if history.shape[1:] != (c.target_dim + c.exog_dim, c.history_patches):
            raise ValueError('History patch grid differs from the saved configuration')
        if not torch.isfinite(history).all():
            raise ValueError('Observed history summaries must be finite')
        return estimate_leads(history[:, :c.target_dim], history[:, c.target_dim:],
            hm[:, :c.target_dim], hm[:, c.target_dim:], c.max_lag, c.min_pairs,
            c.min_correlation, c.leader_num, estimator=c.lag_estimator)

    def base_prediction(self, history, hm, future, fm):
        c = self.config
        regular, mean = fill_history_means(history.transpose(1, 2), hm.transpose(1, 2))
        supplied_future = torch.where(fm.transpose(1, 2), future.transpose(1, 2), mean[..., c.target_dim:])
        self.core.temporal_encoder.criterion.mask = fm.transpose(1, 2)
        self.core.cov_encoder.criterion.mask = hm[:, :c.target_dim].transpose(1, 2)
        temporal, predicted_x, temporal_loss = self.core.temporal_encoder(regular, supplied_future, True)
        # For dag-lift/history, apply DAG's predicted-X inference path in BOTH
        # train and eval. Actual future controls have no bypass around alignment.
        cov_input = supplied_future if c.variant == 'dag' else predicted_x
        covariate, cov_loss = self.core.cov_encoder(regular, cov_input, True)
        base = c.alpha * temporal + (1 - c.alpha) * covariate
        auxiliary = c.beta * (temporal_loss + cov_loss)
        return base.transpose(1, 2), auxiliary, predicted_x

    def forward_grid(self, batch):
        c = self.config
        history, hm, future, fm = self.regularize(batch)
        base, auxiliary, _ = self.base_prediction(history, hm, future, fm)
        if c.variant == 'dag-lift':
            leads = estimate_leads(history[:, :c.target_dim], history[:, c.target_dim:],
                hm[:, :c.target_dim], hm[:, c.target_dim:], c.max_lag, c.min_pairs,
                c.min_correlation, c.leader_num, estimator=c.lag_estimator)
            aligned, available = align_exogenous(history[:, c.target_dim:], future,
                hm[:, c.target_dim:], fm, leads)
            base = self.refiner(history[:, :c.target_dim], hm[:, :c.target_dim],
                base, aligned, available, leads)
        return base.transpose(1, 2), auxiliary

    def forward(self, batch):
        # Labels and future Y masks are never read. Predictions are piecewise
        # constant per fixed-time bin, evaluated at the original query times.
        grid, auxiliary = self.forward_grid(batch)
        indices = batch['query_patch_indices'][..., None].expand(-1, -1, self.config.target_dim)
        return grid.gather(1, indices).contiguous(), auxiliary
