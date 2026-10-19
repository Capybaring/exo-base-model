"""Official LIFT efficient lead estimation, restricted to X -> Y.

The temporal FFT is circular (n=L), and local peaks are searched at 1..L-2,
exactly as the pinned official estimate_indicator default. TTCN feature spectra
are averaged in their shared basis; P=1 reduces to the official scalar formula.
Missing patches are filled with their observed-history feature means before
normalization, so their normalized contribution is zero. No per-lag Pearson
recentring, minimum-pair threshold, or correlation threshold is applied.
"""
import torch


def validate_lead_settings(config):
    if config.lag_estimator not in ('fft', 'pearson'):
        raise ValueError('lag_estimator must be fft or pearson')
    if config.variant not in ('dag-lift', 'patch-lift'):
        return
    if config.lag_estimator == 'fft':
        if config.history_patches < 3:
            raise ValueError('LIFT FFT local peaks require at least 3 history patches')
    elif (config.max_lag < 1 or config.min_pairs < 3 or
          not 0 <= config.min_correlation <= 1 or config.history_patches < config.min_pairs + 1):
        raise ValueError('Pearson requires max_lag >=1, min_pairs >=3, correlation in [0,1], '
                         'and at least min_pairs+1 history patches')


def normalized_history(values, mask):
    valid = mask.bool()[..., None]
    safe = torch.where(valid, values, 0)
    count = valid.sum(-2, keepdim=True).clamp_min(1)
    mean = safe.sum(-2, keepdim=True) / count
    centered = torch.where(valid, safe - mean, 0)
    variance = centered.square().sum((-2, -1), keepdim=True) / (count * values.shape[-1])
    return centered / (variance + 1e-8).sqrt(), variance[..., 0, 0]


@torch.no_grad()
def estimate_fft_leads(y, x, ym, xm, leader_num):
    """Return the common lead diagnostic contract for [B,C,L,P] histories."""
    if y.shape[0] != x.shape[0] or y.shape[-2:] != x.shape[-2:]:
        raise ValueError('Lead inputs must share batch, patch grid and feature basis')
    length, features = y.shape[-2:]
    if length < 3 or leader_num < 1:
        raise ValueError('LIFT FFT local peaks require at least 3 history patches and one leader')
    yn, yv = normalized_history(y, ym)
    xn, xv = normalized_history(x, xm)
    yf = torch.fft.rfft(yn, dim=-2)
    xf = torch.fft.rfft(xn, dim=-2)
    spectrum = torch.einsum('byfp,bxfp->byxf', yf, xf.conj()) / features
    correlation = torch.fft.irfft(spectrum, n=length, dim=-1) / length
    absolute = correlation.abs()
    peaks = ((absolute[..., 1:-1] >= absolute[..., :-2]) &
             (absolute[..., 1:-1] >= absolute[..., 2:]))
    # Keep the official zeroing and first-maximum tie behaviour, including r=0.
    candidates = correlation[..., 1:-1] * peaks
    score, shift = candidates.abs().max(-1)
    corr = candidates.gather(-1, shift[..., None]).squeeze(-1)
    lag = shift + 1

    # Counts describe circular pairs, not the old noncircular overlap. They do
    # not reweight the correlation or impose a configurable count threshold.
    pairs = torch.fft.irfft(torch.fft.rfft(ym.to(y.dtype)).unsqueeze(2) *
                           torch.fft.rfft(xm.to(y.dtype)).conj().unsqueeze(1),
                           n=length)
    counts = pairs.gather(-1, lag[..., None]).squeeze(-1).round().long().clamp(0, length)
    valid = ((yv > 0).unsqueeze(-1) & (xv > 0).unsqueeze(-2) &
             (ym.sum(-1) > 1).unsqueeze(-1) & (xm.sum(-1) > 1).unsqueeze(-2) &
             (counts > 0))
    lag = torch.where(valid, lag, 0)
    corr = torch.where(valid, corr, 0)
    counts = torch.where(valid, counts, 0)
    leaders = score.masked_fill(~valid, -1).topk(min(leader_num, x.shape[1]), -1).indices
    selected = torch.zeros_like(valid).scatter(-1, leaders, True) & valid
    return {'lag': lag, 'correlation': corr, 'pair_count': counts, 'valid': valid,
            'selected': selected, 'leaders': leaders, 'leader_lag': lag.gather(-1, leaders),
            'leader_correlation': corr.gather(-1, leaders),
            'leader_valid': valid.gather(-1, leaders)}


def correlation_features(estimates, temperature):
    """Official [1, abs(r)] / temperature, excluding invalid padding slots."""
    r = estimates['leader_correlation'].abs()
    logits = torch.cat([torch.ones_like(r[..., :1]), r], -1) / temperature
    valid = torch.cat([torch.ones_like(estimates['leader_valid'][..., :1]),
                       estimates['leader_valid']], -1)
    return logits.masked_fill(~valid, -torch.inf).softmax(-1)[..., 1:]
