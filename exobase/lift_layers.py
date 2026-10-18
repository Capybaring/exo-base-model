"""Unchanged LIFT FilterFactory and ComplexLinear classes (79cf8f1).

Only unrelated imports/classes are omitted. See upstream/lift/LIFT.py.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class FilterFactory(nn.Module):
    def __init__(self, feat_dim, seq_len, num_channel, out_dim, num_state=1, need_classifier=True):
        super().__init__()
        self.num_state = num_state
        self.need_classifier = need_classifier
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.out_dim = out_dim
        if num_state > 1 and need_classifier:
            self.classifier = nn.Linear(seq_len, num_state, bias=False)
            self.basic_state = nn.Parameter(torch.empty(num_channel, num_state))
            self.bias = nn.Parameter(torch.empty(num_state))
            bound = 1 / math.sqrt(seq_len)
            nn.init.uniform_(self.bias, -bound, bound)
            bound = 1 / math.sqrt(num_state)
            nn.init.uniform_(self.basic_state, -bound, bound)

        if self.num_state == 1:
            self.mix_head = nn.Linear(feat_dim, self.out_dim)
        else:
            tmp_linear = nn.Linear(feat_dim, self.out_dim * num_state)
            self.mix_head_w = nn.Parameter(tmp_linear.weight.clone().view(self.out_dim, num_state, feat_dim).permute(1, 2, 0).reshape(num_state, -1))
            self.mix_head_b = nn.Parameter(tmp_linear.bias.clone().view(self.out_dim, num_state).transpose(0, 1))

    def forward(self, lookback_seq, corr_feat, p=None):
        if self.num_state == 1:
            return self.mix_head(corr_feat), None
        B, C = corr_feat.shape[:-1]
        if self.need_classifier:
            p = self.bias + self.basic_state + self.classifier(lookback_seq)
            p = torch.softmax(p, -1).unsqueeze(-2) # [B, C, 1, S]
        weight = (p @ self.mix_head_w).view(B, C, self.feat_dim, self.out_dim)
        bias = (p @ self.mix_head_b).view(B, C, self.out_dim)
        r = (corr_feat.unsqueeze(-2) @ weight).squeeze(-2) + bias
        return r, p


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        weight_complex = torch.empty((out_features, in_features), dtype=torch.complex64)
        nn.init.kaiming_uniform_(weight_complex, a=math.sqrt(5))
        self.weight_real = nn.Parameter(weight_complex.real)
        self.weight_imag = nn.Parameter(weight_complex.imag)
        self.bias = bias
        if bias:
            bias_complex = torch.empty(out_features, dtype=torch.complex64)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight_complex)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias_complex, -bound, bound)
            self.bias_real = nn.Parameter(bias_complex.real)
            self.bias_imag = nn.Parameter(bias_complex.imag)
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, x):
        return F.linear(x, torch.view_as_complex(torch.stack([self.weight_real, self.weight_imag], -1)),
                        torch.view_as_complex(torch.stack([self.bias_real, self.bias_imag], -1)) if self.bias else None)
