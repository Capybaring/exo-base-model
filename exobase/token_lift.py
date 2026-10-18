"""Directed, noncircular LIFT adaptation for full TTCN feature sequences.

Time is always the patch axis. Features are never flattened into fake time or
averaged into a scalar. Lead scores are history-only feature-centered Pearson
correlations in the shared TTCN basis, not correlations of raw physical means.
"""
import torch
from torch import nn

from .lift_layers import FilterFactory, ComplexLinear


def token_statistics(values, mask):
    """[B,C,T,P]: per-feature temporal mean and one joint RMS per variable."""
    valid = mask.bool().unsqueeze(-1)
    safe = torch.where(valid, values, 0)
    count = valid.sum(-2, keepdim=True).clamp_min(1)
    mean = safe.sum(-2, keepdim=True)/count
    centered = torch.where(valid, safe-mean, 0)
    variance = centered.square().sum((-2, -1), keepdim=True)/(count*values.shape[-1])
    std = (variance+1e-8).sqrt()
    return safe, mean, std


@torch.no_grad()
def estimate_token_leads(y, x, ym, xm, max_lag, min_pairs, min_correlation, leader_num):
    """[B,Y,L,P]/[B,X,L,P], excluding the TTCN validity feature."""
    if y.shape[-2:] != x.shape[-2:]:
        raise ValueError('Lead inputs must share patch grid and TTCN feature basis')
    y, x = torch.where(ym[..., None], y, 0), torch.where(xm[..., None], x, 0)
    b, ny, length, _ = y.shape
    nx = x.shape[1]
    shape = (b, ny, nx)
    best = y.new_full(shape, -1.)
    corr = y.new_zeros(shape)
    lag = torch.zeros(shape, dtype=torch.long, device=y.device)
    counts = torch.zeros_like(lag)
    for ell in range(1, min(max_lag, length-1)+1):
        xx, yy = x[:, None, :, :-ell], y[:, :, None, ell:]
        valid = xm[:, None, :, :-ell] & ym[:, :, None, ell:]
        n = valid.sum(-1)
        mx = torch.where(valid[..., None], xx, 0).sum(-2)/n.clamp_min(1)[..., None]
        my = torch.where(valid[..., None], yy, 0).sum(-2)/n.clamp_min(1)[..., None]
        xc = torch.where(valid[..., None], xx-mx[..., None, :], 0)
        yc = torch.where(valid[..., None], yy-my[..., None, :], 0)
        vx, vy = xc.square().sum((-2, -1)), yc.square().sum((-2, -1))
        r = (xc*yc).sum((-2, -1))/(vx*vy).sqrt().clamp_min(1e-12)
        eligible = (n >= min_pairs) & (vx > 1e-10) & (vy > 1e-10)
        score = torch.where(eligible, r.abs(), -1.)
        improve = score > best
        best = torch.where(improve, score, best)
        corr = torch.where(improve, r.clamp(-1, 1), corr)
        lag = torch.where(improve, ell, lag)
        counts = torch.where(improve, n, counts)
    valid = (lag > 0) & (corr.abs() >= min_correlation)
    leaders = corr.abs().masked_fill(~valid, -1.).topk(min(leader_num, nx), -1).indices
    selected = torch.zeros_like(valid).scatter(-1, leaders, True) & valid
    return {'lag': lag, 'correlation': corr, 'pair_count': counts, 'valid': valid,
            'selected': selected, 'leaders': leaders, 'leader_lag': lag.gather(-1, leaders),
            'leader_correlation': corr.gather(-1, leaders),
            'leader_valid': valid.gather(-1, leaders)}


def align_token_exogenous(x, future, xm, fm, estimates):
    """X[L+h-lag] in the shared TTCN basis, normalized from observed history."""
    b, nx, length, p = x.shape
    horizon = future.shape[-2]
    _, mean, std = token_statistics(x, xm)
    mask = torch.cat([xm, fm], -1)
    sequence = torch.cat([x, future], -2)
    sequence = torch.where(mask[..., None], (sequence-mean)/std, 0)
    leaders = estimates['leaders']
    ny, k = leaders.shape[1:]
    channels = leaders.reshape(b, -1, 1, 1).expand(-1, -1, length+horizon, p)
    source = sequence.gather(1, channels).reshape(b, ny, k, length+horizon, p)
    observed = mask.gather(1, leaders.reshape(b, -1, 1).expand(-1, -1, length+horizon))
    observed = observed.reshape(b, ny, k, length+horizon)
    index = torch.arange(length, length+horizon, device=x.device)-estimates['leader_lag'][..., None]
    in_range = (index >= 0) & (index < length+horizon)
    index = index.clamp(0, length+horizon-1)
    available = observed.gather(-1, index) & in_range & estimates['leader_valid'][..., None]
    aligned = source.gather(-2, index[..., None].expand(-1, -1, -1, -1, p))
    aligned = aligned*estimates['leader_correlation'].sign()[..., None, None]
    return torch.where(available[..., None], aligned, 0), available


class TokenLeadRefiner(nn.Module):
    def __init__(self, history, horizon, targets, leaders, features, states, mode):
        super().__init__()
        if mode not in ('prefix', 'global'):
            raise ValueError('Unknown token LIFT mode')
        self.horizon, self.k, self.mode = horizon, leaders, mode
        self.temperature = nn.Parameter(torch.ones(()))
        freq = horizon//2+1
        self.factory = FilterFactory(leaders, history*features, targets,
                                     freq*(2*leaders+1), states)
        self.mix_layer = ComplexLinear(3*freq, freq)

    def forward(self, history, mask, base, aligned, available, estimates):
        """base [B,Y,H,P], aligned [B,Y,K,H,P]; FFT along H only."""
        safe, mean, std = token_statistics(history, mask)
        filled = torch.where(mask[..., None], safe, mean)
        normalized = (base-mean)/std
        r = estimates['leader_correlation'].abs()*estimates['leader_valid']
        feature = torch.cat([torch.ones_like(r[..., :1]), r], -1)
        feature = (feature/self.temperature.abs().clamp_min(.05)).softmax(-1)[..., 1:]
        filters, _ = self.factory(filled.flatten(-2), feature)
        filters = filters.reshape(*base.shape[:2], 2*self.k+1, -1)
        if self.mode == 'prefix':
            prefix = torch.ones(self.horizon, self.horizon, dtype=torch.bool,
                                device=base.device).tril()
            controls = aligned.unsqueeze(2)*prefix[None, None, :, None, :, None]
            filters = filters.unsqueeze(2)
            yf = torch.fft.rfft(normalized, dim=-2).unsqueeze(2)
            active = (available.unsqueeze(2) & prefix[None, None, :, None, :]).any((-1, -2))
        else:
            controls = aligned
            yf = torch.fft.rfft(normalized, dim=-2)
            active = available.any(-2)
        uf = torch.fft.rfft(controls, dim=-2)
        filtered = uf*filters[..., :self.k, :, None]
        difference = (filtered-yf.unsqueeze(-3))*filters[..., self.k:-1, :, None]
        own = (yf*filters[..., -1, :, None]).expand_as(filtered.sum(-3))
        streams = torch.cat([filtered.sum(-3), difference.sum(-3), own], -2)
        spectrum = self.mix_layer(streams.transpose(-1, -2)).transpose(-1, -2)
        delta = torch.fft.irfft(spectrum, n=self.horizon, dim=-2)
        if self.mode == 'prefix':
            delta = delta.diagonal(dim1=2, dim2=3).transpose(-1, -2)
        return base+torch.where(active[..., None], delta*std, 0)
