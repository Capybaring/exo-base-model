"""Adapted from Capybaring/dag f446890; see upstream/dag and docs/PROVENANCE.md."""
import numpy as np
import torch
from torch import nn
from math import sqrt

class TriangularCausalMask:
    def __init__(self, batch, length, device):
        self.mask = torch.ones(batch, 1, length, length, dtype=torch.bool, device=device).triu(1)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, exog_attn=None, attn_alpha=0.5, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A_before_dropout = torch.softmax(scale * scores, dim=-1)
        if exog_attn is not None:
            # A_before_dropout = (1.0 - attn_alpha) * A_before_dropout + attn_alpha * exog_attn TODO
            A_before_dropout = A_before_dropout + attn_alpha * exog_attn
            A_before_dropout = torch.softmax(A_before_dropout, dim=-1)
        A = self.dropout(A_before_dropout)

        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A_before_dropout
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, exog_attn=None, attn_alpha=0.5, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            exog_attn,
            tau=tau,
            delta=delta,
            attn_alpha=attn_alpha
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
