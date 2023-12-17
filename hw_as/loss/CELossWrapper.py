import torch.nn as nn
import torch

class CELossWrapper(nn.CrossEntropyLoss):
    def __init__(self, weight=None):
        if weight is not None:
            weight = torch.tensor(weight)
        super().__init__(weight=weight)

    def forward(self, logits, target, **batch):
        return super().forward(logits, target)
