### Spoof detection
## Overview
This repository provides code for detecting generated audios. The model used is AASIST.

## Install
To install used packages run `pip install -r requiremets.txt`. Torch-geometric sometimes downloads incorrect. In that case try running
```
pip install -q torch-scatter -f https://data.pyg.org/whl/torch-{torchversion}.html
pip install -q torch-sparse -f https://data.pyg.org/whl/torch-{torchversion}.html
pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
```
The necessary torchversion could be find using
```
import torch
torchversion = torch.__version__
print(torchversion)
```

## Train
Run `python train.py -c your_config_path -r checkpoint_path` to train AASIST model with config located in `your_config_path`. If you want to resume training from existing checkpoint, add `checkpoint_path`. New model architecture must match the one used in previous training.

## Evaluation
To get predictions for desired audios, put them in `test_audios` folder and run `python test.py`. Paths to the audios and obtained score will be printed to standard output.

## Dataset
The dataset used for training is LA partition of ASVspoof 2019 Dataset. To download it run `download_dataset.sh` script. It requires your kaggle api token to be located in `~/.kaggle/kaggle.json`.

## Results
test\_audios/audio\_3.flac: bonafide prob: 0.03712696582078934
test\_audios/audio\_1.flac: bonafide prob: 0.03034604713320732
test\_audios/generated.mp3: bonafide prob: 0.010173882357776165
test\_audios/audio\_3.flac: bonafide prob: 7.031096902210265e-05
test\_audios/unknown.mp3: bonafide prob: 0.8339422345161438

