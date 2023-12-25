from math import floor, ceil
import torch.nn.functional as F
import torch
from random import randint


def truncate(x):
    full_t = 64600 // x.shape[-1]
    tmp = torch.tile(x, (1,full_t))
    rest = 64600 % x.shape[-1]
    x = torch.cat([tmp, x[:,:rest]], dim=-1)
    idx = randint(64600, x.shape[-1])
    return x[..., idx - 64600:idx]

def collate_fn(batch):
    x = torch.cat([truncate(x) for x, _, _ in batch], dim=0)
    y = torch.tensor([y for _, y, _ in batch])
    paths = [path for _, _, path in batch]
    return {"audio": x, "target": y, "audio_path": paths}
