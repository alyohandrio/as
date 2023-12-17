import torch
import torch.nn as nn
from hw_as.modules import AASISTEncoder, GraphModule, MGO


class AASIST(nn.Module):
    def __init__(self, sinc_channels, sinc_kernel, res_channels, gat_ks, gat_dropout, pool_dropout, mgo_in_dims, mgo_out_dims, mgo_ks, feat_dropout, final_dropout):
        super().__init__()
        self.encoder = AASISTEncoder(sinc_channels, sinc_kernel, res_channels)
        self.spec_pe = nn.Parameter(torch.randn(1, 23, res_channels[-1][-1]))
        self.graph_module1 = GraphModule(res_channels[-1][-1], mgo_in_dims[0], gat_ks[0], gat_dropout, pool_dropout)
        self.graph_module2 = GraphModule(res_channels[-1][-1], mgo_in_dims[0], gat_ks[1], gat_dropout, pool_dropout)
        self.mgo = MGO(mgo_in_dims, mgo_out_dims, mgo_ks, gat_dropout, pool_dropout, feat_dropout, final_dropout)

    def forward(self, audio, **batch):
        x = self.encoder(audio)
        xs = torch.max(x, dim=3).values.transpose(1, 2) + self.spec_pe
        xs = self.graph_module1(xs)
        xt = torch.max(x, dim=2).values.transpose(1, 2)
        xt = self.graph_module2(xt)
        return self.mgo(xs, xt)
