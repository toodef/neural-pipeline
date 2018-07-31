import torch

eps = 1e-3


def dice_loss(preds, trues):
    preds = preds.contiguous()
    trues = trues.contiguous()
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    intersection = (preds * trues).sum(1)
    scores = (2. * intersection + eps) / (preds.sum(1) + trues.sum(1) + eps)

    return float(torch.clamp(scores.sum(), 0., 1.))


def jaccard(preds, trues):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    intersection = (preds * trues).sum(1)
    scores = (intersection + eps) / ((preds + trues).sum(1) - intersection + eps)

    return float(torch.clamp(scores.sum(), 0., 1.))
