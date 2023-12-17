from torch.utils.data import Dataset
import torchaudio
import os

class ASVDataset(Dataset):
    def __init__(self, root, split, l_suf, **batch):
      super().__init__()
      self.root = root
      self.split = split
      labels_path = os.path.join(root, f"LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{l_suf}.txt")
      data_path = os.path.join(root, f"LA/LA/ASVspoof2019_LA_{split}/flac")
      with open(labels_path, 'r') as f:
          self.labels = list(map(lambda x: (x.split()[1], 0 if x.split()[4] == "bonafide" else 1), f.readlines()))

    def __getitem__(self, idx):
        path, label = self.labels[idx]
        path = os.path.join(self.root, f"LA/LA/ASVspoof2019_LA_{self.split}/flac", path + ".flac")
        waveform, sr = torchaudio.load(path)
        return waveform, label, path
    
    def __len__(self):
        return len(self.labels)
