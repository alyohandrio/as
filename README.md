### Spoof detection
This repository provides code for detecting generated audios. To use it put your tested files in test\_audios folder. The model used is AASIST.

To install used packages run `pip install -r requiremets.txt`. Torch-geometric sometimes downloads incorrect. In that case try running
`
pip install -q torch-scatter -f https://data.pyg.org/whl/torch-{torchversion}.html
pip install -q torch-sparse -f https://data.pyg.org/whl/torch-{torchversion}.html
pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
`

The necessary torchversion could be find using
`
import torch
torchversion = torch.__version__
print(torchversion)

Results:
test_audios/audio_3.flac: bonafide prob: 0.032531432807445526
test_audios/audio_1.flac: bonafide prob: 0.0006950827082619071
test_audios/generated.mp3: bonafide prob: 0.007867229171097279
test_audios/audio_2.flac: bonafide prob: 0.000498166074976325
test_audios/unknown.mp3: bonafide prob: 0.3997294008731842
`
