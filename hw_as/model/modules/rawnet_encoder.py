import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from hw_as.model.modules.sinc_conv import SincConv_fast


class ResidualBlockWithFMS(nn.Module):
    def __init__(self, in_channels, out_channels, slope):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(slope),
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(slope),
            nn.Conv1d(out_channels, out_channels, 3, padding=1)
        )
        if in_channels != out_channels:
            self.ds_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.ds_conv = nn.Identity()
        self.pooling = nn.MaxPool1d(3)
        self.lin = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        out = self.body(x) + self.ds_conv(x)
        out = self.pooling(out)
        s = self.lin(out.mean(dim=-1))
        s = F.sigmoid(s).unsqueeze(-1)
        out = out * s + s
        return out


class RawNetEncoder(nn.Module):
    def __init__(self, sinc_channels=128, sinc_kernel=129, res_channels=[[128, 128], [128, 128], [128, 512], [512, 512], [512, 512], [512, 512]], slope=0.2, min_low_hz=0, min_band_hz=0):
        super().__init__()
        self.sinc = SincConv_fast(sinc_channels, sinc_kernel, min_low_hz=min_low_hz, min_band_hz=min_band_hz)
        self.post_sinc = nn.Sequential(
            nn.MaxPool1d(3),
            nn.BatchNorm1d(sinc_channels),
            nn.LeakyReLU(slope)
        )
        self.res_blocks = nn.ModuleList([ResidualBlockWithFMS(channels[0], channels[1], slope) for channels in res_channels])

    def forward(self, x):
        x = self.sinc(x.unsqueeze(1))
        x = torch.abs(x)
        x = self.post_sinc(x)
        for block in self.res_blocks:
            x = block(x)
        return x
