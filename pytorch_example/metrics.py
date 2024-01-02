"""
Taken from Andrew McDonald's https://github.com/ampersandmcd/icenet-gan/blob/main/src/metrics.py
Adapted from Tom Andersson's https://github.com/tom-andersson/icenet-paper/blob/main/icenet/metrics.py
Modified from Tensorflow to PyTorch and PyTorch Lightning.
Extended to include additional sharpness metrics.
"""
import torch
from torchmetrics import Metric


class IceNetAccuracy(Metric):
    """
    Binary accuracy metric for use at multiple leadtimes.
    """    

    # Set class properties
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True

    def __init__(self, leadtimes_to_evaluate: list):
        """
        Construct a binary accuracy metric for use at multiple leadtimes.
        :param leadtimes_to_evaluate: A list of leadtimes to consider
            e.g., [0, 1, 2, 3, 4, 5] to consider all six months in accuracy computation or
            e.g., [0] to only look at the first month's accuracy
            e.g., [5] to only look at the sixth month's accuracy
        """
        super().__init__()
        self.leadtimes_to_evaluate = leadtimes_to_evaluate
        self.add_state("weighted_score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("possible_score", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor):
        # preds and target are shape (b, h, w, t)
        # sum marginal and full ice for binary eval
        preds = (preds > 0).long()
        target = (target > 0).long()
        base_score = preds[:, :, :, self.leadtimes_to_evaluate] == target[:, :, :, self.leadtimes_to_evaluate]
        self.weighted_score += torch.sum(base_score * sample_weight[:, :, :, self.leadtimes_to_evaluate])
        self.possible_score += torch.sum(sample_weight[:, :, :, self.leadtimes_to_evaluate])

    def compute(self):
        return self.weighted_score.float() / self.possible_score


class SIEError(Metric):
    """
    Sea Ice Extent error metric (in km^2) for use at multiple leadtimes.
    """ 

    # Set class properties
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = True

    def __init__(self, leadtimes_to_evaluate: list):
        """
        Construct an SIE error metric (in km^2) for use at multiple leadtimes.
        :param leadtimes_to_evaluate: A list of leadtimes to consider
            e.g., [0, 1, 2, 3, 4, 5] to consider all six months in computation or
            e.g., [0] to only look at the first month
            e.g., [5] to only look at the sixth month
        """
        super().__init__()
        self.leadtimes_to_evaluate = leadtimes_to_evaluate
        self.add_state("pred_sie", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("true_sie", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor):
        # preds and target are shape (b, h, w, t)
        # sum marginal and full ice for binary eval
        preds = (preds > 0).long()
        target = (target > 0).long()
        self.pred_sie += preds[:, :, :, self.leadtimes_to_evaluate].sum()
        self.true_sie += target[:, :, :, self.leadtimes_to_evaluate].sum()

    def compute(self):
        return (self.pred_sie - self.true_sie) * 25**2 # each pixel is 25x25 km