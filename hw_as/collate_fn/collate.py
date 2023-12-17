from math import floor, ceil
import torch.nn.functional as F
import torch
from random import randint


def truncate(x):
    dif = 64600 - x.shape[-1]
    dif = max(dif, 0)
    x = F.pad(x, (floor(dif / 2), ceil(dif / 2)))
    idx = randint(64600, x.shape[-1])
    return x[..., idx - 64600:idx]


def collate_fn(batch):
    x = torch.cat([truncate(x) for x, _, _ in batch], dim=0)
    y = torch.tensor([y for _, y, _ in batch])
    paths = [path for _, _, path in batch]
    return {"audio": x, "target": y, "audio_path": paths}
