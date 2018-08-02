import torch
import numpy as np

eps = 1e-3


def dice_loss(preds, trues):
    preds = preds.contiguous()
    trues = trues.contiguous()
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    intersection = (preds * trues).sum(1)
    scores = (2. * intersection + eps) / (preds.sum(1) + trues.sum(1) + eps)

    return float(scores.sum())


def jaccard(preds, trues):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    intersection = (preds * trues).sum(1)
    scores = (intersection + eps) / ((preds + trues).sum(1) - intersection + eps)

    return float(scores.sum())


def masked_jaccard(preds, trues, threshold):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)

    preds = preds.data.cpu().numpy()
    trues = trues.data.cpu().numpy()

    preds_mask = np.zeros((preds.shape[0], preds.shape[1]), dtype=np.int)
    preds_mask[preds > threshold] = 1
    trues_mask = np.zeros((preds.shape[0], preds.shape[1]), dtype=np.int)
    trues_mask[trues > 0] = 1

    intersection_mask = np.bitwise_and(trues_mask, preds_mask)
    union_mask = np.bitwise_or(trues_mask, preds_mask)

    scores = intersection_mask.sum(1) / (union_mask.sum(1) + eps)

    return float(scores.sum())
