"""Residual dilated causal TCN, trained from scratch on provided forecasts."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel, dilation):
        super().__init__()
        self.left_padding = (kernel-1)*dilation
        self.conv = nn.Conv1d(channels_in, channels_out, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, kernel, dilation, dropout):
        super().__init__()
        self.layers = nn.Sequential(CausalConv(cin, cout, kernel, dilation), nn.GELU(), nn.Dropout(dropout),
                                    CausalConv(cout, cout, kernel, dilation), nn.GELU(), nn.Dropout(dropout))
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return F.gelu(self.layers(x)+self.skip(x))


class ForecastTCN(nn.Module):
    def __init__(self, input_size, config, nsites=253):
        super().__init__()
        self.embedding = nn.Embedding(nsites, config["embedding_dim"])
        blocks = []
        cin = input_size + config["embedding_dim"]
        channels = config["channels"]
        for level in range(config["levels"]):
            blocks.append(ResidualBlock(cin, channels, config["kernel_size"], 2**level, config["dropout"]))
            cin = channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Conv1d(channels, 1, 1)
        nn.init.constant_(self.head.bias, -2.)

    def forward(self, x, site):
        embedding = self.embedding(site).unsqueeze(1).expand(-1, x.shape[1], -1)
        z = torch.cat([x, embedding], dim=-1).transpose(1, 2)
        return F.softplus(self.head(self.network(z)).squeeze(1))


def receptive_field(config):
    return 1+2*(config["kernel_size"]-1)*(2**config["levels"]-1)
