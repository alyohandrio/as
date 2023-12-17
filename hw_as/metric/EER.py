import torch.nn.functional as F
from hw_as.metric.calculate_eer import compute_eer

class EER:
    def __init__(self, name=None):
        self.name = name if name is not None else type(self).__name__

    def __call__(self, logits, target, **batch):
        probs = F.softmax(logits, dim=-1)
        return compute_eer(probs[target == 0][:,0].detach().cpu().numpy(), probs[target == 1][:,0].detach().cpu().numpy())[0]
