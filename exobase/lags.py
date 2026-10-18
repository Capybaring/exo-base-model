"""History-only X->Y lead estimation and known-exogenous LIFT refinement.

This is an explicit adaptation, not the original circular FFT lead estimator:
Pearson correlations use only finite, observed, overlapping history pairs.
LIFT's state-dependent filter factory and three frequency streams are retained.
Prefix mode prevents later aligned controls from affecting earlier predictions.
"""
import torch
from torch import nn

from .lift_layers import FilterFactory, ComplexLinear


def masked_statistics(values, mask):
    mask = mask.bool()
    safe = torch.where(mask, values, torch.zeros_like(values))
    count = mask.sum(-1, keepdim=True)
    mean = safe.sum(-1, keepdim=True) / count.clamp_min(1)
    centered = torch.where(mask, safe - mean, torch.zeros_like(safe))
    std = (centered.square().sum(-1, keepdim=True) / count.clamp_min(1) + 1e-8).sqrt()
    return safe, mean, std


@torch.no_grad()
def estimate_leads(y, x, ym, xm, max_lag=48, min_pairs=8,
                   min_correlation=.3, leader_num=3):
    """Inputs [B,Y,L]/[B,X,L]; positive ell means X[s-ell] predicts Y[s].

    No wraparound, interpolation, future samples or future target masks enter
    this estimator. An ineligible pair has lag=0, valid=False (not zero delay).
    Returned leader indices always index X, never the concatenated [Y,X] set.
    """
    y = torch.where(ym, y, 0).detach()
    x = torch.where(xm, x, 0).detach()
    b, ny, length = y.shape
    nx = x.shape[1]
    shape = (b, ny, nx)
    best = y.new_full(shape, -1.)
    corr = y.new_zeros(shape)
    lag = torch.zeros(shape, device=y.device, dtype=torch.long)
    counts = torch.zeros_like(lag)
    for ell in range(1, min(max_lag, length - 1) + 1):
        xx, yy = x[:, None, :, :-ell], y[:, :, None, ell:]
        valid = xm[:, None, :, :-ell] & ym[:, :, None, ell:]
        n = valid.sum(-1)
        mx = torch.where(valid, xx, 0).sum(-1) / n.clamp_min(1)
        my = torch.where(valid, yy, 0).sum(-1) / n.clamp_min(1)
        xc = torch.where(valid, xx - mx[..., None], 0)
        yc = torch.where(valid, yy - my[..., None], 0)
        vx, vy = xc.square().sum(-1), yc.square().sum(-1)
        r = (xc * yc).sum(-1) / (vx * vy).sqrt().clamp_min(1e-12)
        eligible = (n >= min_pairs) & (vx > 1e-10) & (vy > 1e-10)
        score = torch.where(eligible, r.abs(), -1.)
        improve = score > best  # exact ties prefer the shorter lag
        best = torch.where(improve, score, best)
        corr = torch.where(improve, r.clamp(-1, 1), corr)
        lag = torch.where(improve, ell, lag)
        counts = torch.where(improve, n, counts)
    pair_valid = (lag > 0) & (corr.abs() >= min_correlation)
    k = min(leader_num, nx)
    scores = corr.abs().masked_fill(~pair_valid, -1.)
    leaders = scores.topk(k, dim=-1).indices
    selected = torch.zeros_like(pair_valid).scatter(-1, leaders, True) & pair_valid
    return {'lag': lag, 'correlation': corr, 'pair_count': counts,
            'valid': pair_valid, 'selected': selected, 'leaders': leaders,
            'leader_lag': lag.gather(-1, leaders),
            'leader_correlation': corr.gather(-1, leaders),
            'leader_valid': pair_valid.gather(-1, leaders)}


def align_exogenous(x, future_x, xm, future_mask, estimates):
    """Normalize X using history alone, then gather X[L+h-delta_ji]."""
    b, nx, length = x.shape
    horizon = future_x.shape[-1]
    _, mean, std = masked_statistics(x, xm)
    mask = torch.cat([xm, future_mask], -1)
    sequence = torch.cat([x, future_x], -1)
    sequence = torch.where(mask, (sequence - mean) / std, 0)
    leaders = estimates['leaders']
    ny, k = leaders.shape[1:]
    channels = leaders.reshape(b, -1, 1).expand(-1, -1, length + horizon)
    source = sequence.gather(1, channels).reshape(b, ny, k, -1)
    observed = mask.gather(1, channels).reshape(b, ny, k, -1)
    indices = (torch.arange(length, length + horizon, device=x.device)
               - estimates['leader_lag'][..., None])
    in_range = (indices >= 0) & (indices < length + horizon)
    indices = indices.clamp(0, length + horizon - 1)
    available = observed.gather(-1, indices) & in_range & estimates['leader_valid'][..., None]
    aligned = source.gather(-1, indices) * estimates['leader_correlation'].sign()[..., None]
    return torch.where(available, aligned, 0), available


class ExogenousLeadRefiner(nn.Module):
    def __init__(self, history, horizon, targets, leaders, states=8, mode='prefix'):
        super().__init__()
        if mode not in ('prefix', 'global'):
            raise ValueError('refiner_mode must be prefix or global')
        self.mode, self.horizon, self.k = mode, horizon, leaders
        self.temperature = nn.Parameter(torch.ones(()))
        freq = horizon // 2 + 1
        self.factory = FilterFactory(leaders, history, targets, freq * (2 * leaders + 1), states)
        # Separate real/imaginary parameters avoid experimental complex optimizer
        # and state-serialization paths; the class is from official LIFT.
        self.mix_layer = ComplexLinear(3 * freq, freq)

    def forward(self, y_history, y_mask, base, aligned, available, estimates):
        """base [B,Y,H], aligned [B,Y,K,H]; never takes future Y."""
        safe, mean, std = masked_statistics(y_history, y_mask)
        filled_history = torch.where(y_mask, safe, mean)
        baseline = (base - mean) / std
        r = estimates['leader_correlation'].abs() * estimates['leader_valid']
        temperature = self.temperature.abs().clamp_min(.05)
        features = torch.cat([torch.ones_like(r[..., :1]), r], -1)
        features = (features / temperature).softmax(-1)[..., 1:]
        filters, _ = self.factory(filled_history, features)
        filters = filters.reshape(*base.shape[:2], 2 * self.k + 1, -1)
        if self.mode == 'prefix':
            prefix = torch.ones(self.horizon, self.horizon, device=base.device,
                                dtype=torch.bool).tril()
            # Each forecast bin gets its own prefix of already aligned controls.
            # Full-horizon frequency mixing otherwise violates this time bound.
            controls = aligned.unsqueeze(2) * prefix[None, None, :, None, :]
            filters = filters.unsqueeze(2)
            yf = torch.fft.rfft(baseline).unsqueeze(2)
            uf = torch.fft.rfft(controls)
            active = (available.unsqueeze(2) & prefix[None, None, :, None, :]).any((-1, -2))
        else:
            yf = torch.fft.rfft(baseline)
            uf = torch.fft.rfft(aligned)
            active = available.any(-2)
        filtered = uf * filters[..., :self.k, :]
        difference = (filtered - yf.unsqueeze(-2)) * filters[..., self.k:-1, :]
        own = (yf * filters[..., -1, :]).expand_as(filtered.sum(-2))
        mixed = torch.cat([filtered.sum(-2), difference.sum(-2), own], -1)
        delta = torch.fft.irfft(self.mix_layer(mixed), n=self.horizon)
        if self.mode == 'prefix':
            delta = delta.diagonal(dim1=-2, dim2=-1)
        return base + torch.where(active, delta * std, 0)
