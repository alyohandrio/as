import json
from hw_as.model import AASIST
from hw_as.collate_fn import collate_fn
import os
import torchaudio
import torch

with open("hw_as/configs/aasist_config.json", 'r') as f:
    cfg = json.load(f)
model = AASIST(**cfg["arch"]["args"])

paths = os.listdir("test_audios")
audios = []
for path in paths:
    pth = os.path.join("test_audios", path)
    waveform, sr = torchaudio.load(pth)
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    audios.append((waveform, -1, pth))
data = collate_fn(audios)
model.load_state_dict(torch.load("checkpoint.pth")["state_dict"])
preds = model(**data)
preds = torch.nn.functional.softmax(preds, dim=-1)
for path, result in zip(data["audio_path"], preds):
    print(f"{path}: bonafide prob: {result[0].item()}")

