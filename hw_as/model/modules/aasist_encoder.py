import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from hw_as.model.modules.sinc_conv import SincConv_fast


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SELU(),
            nn.Conv2d(in_channels, out_channels, (2, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.SELU(),
            nn.Conv2d(out_channels, out_channels, (2, 3), padding=(0, 1))
        )
        if in_channels != out_channels:
            self.ds_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.ds_conv = nn.Identity()
        self.pooling = nn.MaxPool2d((1, 3))

    def forward(self, x):
        out = self.body(x) + self.ds_conv(x)
        out = self.pooling(out)
        return out


class AASISTEncoder(nn.Module):
    def __init__(self, sinc_channels=70, sinc_kernel=128, res_channels=[[1, 32], [32, 32], [32, 64], [64, 64], [64, 64], [64, 64]]):
        super().__init__()
        self.sinc = SincConv_fast(sinc_channels, sinc_kernel)
        self.post_sinc = nn.Sequential(
            nn.MaxPool2d(3),
            nn.BatchNorm2d(1),
            nn.SELU()
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(channels[0], channels[1]) for channels in res_channels])

    def forward(self, x):
        x = self.sinc(x.unsqueeze(1))
        x = self.post_sinc(x.unsqueeze(1).abs())
        for block in self.res_blocks:
            x = block(x)
        return x
