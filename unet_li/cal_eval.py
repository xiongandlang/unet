import torch.nn as nn
import torch.nn.functional as F


def Cal_iou_para(pred, label):
    tp, tn, fp, fn = 0, 0, 0, 0
    pred = F.sigmoid(pred)
    tp = ((pred * label).sum()).sum()
    tn = (((1 - pred) * (1 - label)).sum()).sum()
    fp = ((pred * (1 - label)).sum()).sum()
    fn = (((1 - pred) * label).sum()).sum()
    iou = tp / (tp + fp + fn + 1e-08)
    precise = tp / (tp + fp + 1e-08)
    recall = tp / (tp + fn + 1e-08)
    f1 = 2 * precise * recall / (precise + recall + 1e-08)
    return iou, f1, [tp, tn, fp, fn]
