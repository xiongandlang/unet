import torch.nn.functional as F
import torch.nn as nn


class Dice:
    def __init__(self):
        self.loss = nn.BCEWithLogitsLoss()
        self.weights = 0.2

    def __call__(self, pred, label):
        bce = self.loss(pred, label)
        pred = F.sigmoid(pred)
        tp = (pred * label).sum().sum()
        fp = (pred * (1 - label)).sum().sum()
        fn = ((1 - pred) * label).sum().sum()
        dice = 1 - 2 * tp / (2 * tp + fp + fn)
        return dice * (1 - self.weights) + bce * self.weights
