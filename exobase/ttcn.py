"""Official t-PatchGNN point/time TTCN, isolated from its Transformer/GNN.

The nonempty-patch equations and parameter shapes match the pinned upstream
LearnableTE/TTCN. Masked padding and all-empty patches are explicitly safe.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class TTCNPatchEncoder(nn.Module):
    def __init__(self, dim, te_dim, execution='chunked'):
        super().__init__()
        if dim < 2 or te_dim < 2:
            raise ValueError('TTCN requires dim >=2 and te_dim >=2')
        if execution not in ('chunked', 'full'):
            raise ValueError('TTCN execution must be chunked or full')
        self.execution = execution
        self.dim, self.te_dim = dim, te_dim
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, te_dim-1)
        self.Filter_Generators = nn.Sequential(
            nn.Linear(1+te_dim, dim-1), nn.ReLU(),
            nn.Linear(dim-1, dim-1), nn.ReLU(),
            nn.Linear(dim-1, (1+te_dim)*(dim-1)))
        self.T_bias = nn.Parameter(torch.randn(1, dim-1))

    def time_embedding(self, time):
        return torch.cat([self.te_scale(time), self.te_periodic(time).sin()], -1)

    def forward(self, values, times, mask):
        """[B,K,P,C] observations -> [B,C,K,D] tokens, [B,C,K] validity."""
        if values.shape != times.shape or values.shape != mask.shape or values.ndim != 4:
            raise ValueError('TTCN inputs must share [B,K,P,C]')
        b, k, p, c = values.shape
        valid = mask.permute(0, 3, 1, 2).reshape(-1, p, 1).bool()
        v = values.permute(0, 3, 1, 2).reshape(-1, p, 1)
        t = times.permute(0, 3, 1, 2).reshape(-1, p, 1)
        v, t = torch.where(valid, v, 0), torch.where(valid, t, 0)
        if not torch.isfinite(v).all() or not torch.isfinite(t).all():
            raise ValueError('Nonfinite observed TTCN value or timestamp')
        present = valid.any(1)
        if self.execution == 'full':
            # Official-style scheduling: encode all B*C*K patches in one call,
            # without TTCN gradient checkpointing. Keep the existing safe mask
            # handling and broadcast aggregation, including empty patches.
            tokens = self._encode_points(v, t, valid)
            return tokens.reshape(b, c, k, self.dim), present.reshape(b, c, k)
        # Hour-long DIST patches can contain thousands of points. Recompute
        # small patch groups during backward instead of retaining all filters.
        encoded = []
        for start in range(0, len(v), 16):
            vv, tt, mm = v[start:start+16], t[start:start+16], valid[start:start+16]
            last = (mm.squeeze(-1)*torch.arange(1, p+1, device=mm.device)).amax().item()
            length = max(int(last), 1)
            arguments = (vv[:, :length], tt[:, :length], mm[:, :length])
            if self.training and torch.is_grad_enabled():
                encoded.append(checkpoint(self._encode_points, *arguments, use_reentrant=False))
            else:
                encoded.append(self._encode_points(*arguments))
        tokens = torch.cat(encoded, 0)
        return tokens.reshape(b, c, k, self.dim), present.reshape(b, c, k)

    def _encode_points(self, v, t, valid):
        p = v.shape[1]
        features = torch.cat([v, self.time_embedding(t)], -1)
        logits = self.Filter_Generators(features)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = logits.softmax(1)*valid
        weights = weights/weights.sum(1, keepdim=True).clamp_min(1e-12)
        weights = weights.reshape(-1, p, self.dim-1, 1+self.te_dim)
        pooled = (features.unsqueeze(-2)*weights).sum((1, 3))
        present = valid.any(1)
        hidden = (pooled+self.T_bias).relu()*present
        tokens = torch.cat([hidden, present.to(hidden.dtype)], -1)
        return tokens


class QueryDecoder(nn.Module):
    """Token + real query-time embedding -> scalar at that query timestamp."""
    def __init__(self, dim, te_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim+te_dim, dim), nn.ReLU(),
                                 nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, features, time_embedding):
        # Forecast features omit the observed-validity bit. Predictions exist
        # for all grid bins; append one before the official-style time decoder.
        tokens = torch.cat([features, torch.ones_like(features[..., :1])], -1)
        return self.net(torch.cat([tokens, time_embedding], -1)).squeeze(-1)
