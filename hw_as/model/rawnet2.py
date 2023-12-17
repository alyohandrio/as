import torch
import torch.nn as nn
import torch.nn.functional as F
from hw_as.modules import RawNetEncoder

class RawNet2(nn.Module):
    def __init__(self, sinc_channels, sinc_kernel, res_channels, slope, gru_layers, gru_hidden, fc_out, min_low_hz=0, min_band_hz=0):
        super().__init__()
        self.encoder = RawNetEncoder(sinc_channels, sinc_kernel, res_channels, slope, min_low_hz, min_band_hz)
        self.bn = nn.BatchNorm1d(res_channels[-1][-1])
        self.act = nn.LeakyReLU(slope)
        self.gru = nn.GRU(res_channels[-1][-1], gru_hidden, gru_layers, batch_first=True)
        self.fc = nn.Linear(gru_hidden, fc_out)
        self.head = nn.Linear(fc_out, 2)

    def forward(self, audio, **batch):
        x = self.encoder(audio)
        x = self.act(self.bn(x))
        x, _ = self.gru(x.transpose(1, 2))
        x = self.fc(x[:,-1])
        x = self.head(x)
        return x
