"""
There defined some common metrics
"""

import cv2

import numpy as np
from sklearn.metrics import jaccard_similarity_score

eps = 1e-6


def dice_loss(preds, trues):
    preds_inner = preds.data.cpu().numpy().copy()
    trues_inner = trues.data.cpu().numpy().copy()

    preds_inner = np.reshape(preds_inner, (preds_inner.shape[0], preds_inner.shape[2] * preds_inner.shape[3]))
    trues_inner = np.reshape(trues_inner, (trues_inner.shape[0], trues_inner.shape[2] * trues_inner.shape[3]))

    intersection = (preds_inner * trues_inner).sum(1)
    scores = (2. * intersection + eps) / (preds_inner.sum(1) + trues_inner.sum(1) + eps)

    return scores


def jaccard(preds, trues):
    preds_inner = np.squeeze(preds.cpu().data.numpy().copy(), axis=1)
    trues_inner = np.squeeze(trues.cpu().data.numpy().copy(), axis=1)

    preds_inner = np.reshape(preds_inner, (preds_inner.shape[0], preds_inner.shape[1] * preds_inner.shape[2]))
    trues_inner = np.reshape(trues_inner, (trues_inner.shape[0], trues_inner.shape[1] * trues_inner.shape[2]))
    intersection = (preds_inner * trues_inner).sum(1)
    scores = (intersection + eps) / ((preds_inner + trues_inner).sum(1) - intersection + eps)

    return scores


def masked_jaccard(preds, trues, threshold):
    preds_inner = np.squeeze(preds.data.cpu().numpy().copy(), axis=1)
    trues_inner = np.squeeze(trues.data.cpu().numpy().copy(), axis=1)

    preds_inner[preds_inner < threshold] = 0
    preds_inner[preds_inner >= threshold] = 1

    preds_inner = np.reshape(preds_inner, (preds_inner.shape[0], preds_inner.shape[1] * preds_inner.shape[2])).astype(np.uint8)
    trues_inner = np.reshape(trues_inner, (trues_inner.shape[0], trues_inner.shape[1] * trues_inner.shape[2])).astype(np.uint8)

    intersection_mask = np.bitwise_and(trues_inner, preds_inner).astype(np.float)
    union_mask = np.bitwise_or(trues_inner, preds_inner).astype(np.float)

    scores = intersection_mask.sum(1) / (union_mask.sum(1) + eps)

    return scores


# def iou(preds, trues, threshold=None):
#     def fbeta(threshold_shift=0, beta=2):
#         y_pred_bin = np.round(preds_inner + threshold_shift)
#
#         tp = np.sum(np.round(trues_inner * y_pred_bin), axis=1) + eps
#         fp = np.sum(np.round(np.clip(y_pred_bin - trues_inner, 0, 1)), axis=1)
#         fn = np.sum(np.round(np.clip(trues_inner - preds_inner, 0, 1)), axis=1)
#
#         precision = tp / (tp + fp)
#         recall = tp / (tp + fn)
#
#         beta_squared = beta ** 2
#         return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall)
#
#     preds_inner = np.squeeze(preds.data.cpu().numpy().copy(), axis=1)
#     trues_inner = np.squeeze(trues.data.cpu().numpy().copy(), axis=1)
#     preds_inner = np.reshape(preds_inner, (preds_inner.shape[0], preds_inner.shape[1] * preds_inner.shape[2]))
#     trues_inner = np.reshape(trues_inner, (trues_inner.shape[0], trues_inner.shape[1] * trues_inner.shape[2]))
#     preds_inner = np.clip(preds_inner, 0, 1)
#
#     if threshold is not None:
#         preds_inner[preds_inner < threshold] = 0
#         preds_inner[preds_inner >= threshold] = 1
#
#     thresholds = np.arange(0.5, 0.96, 0.05)
#     return np.sum([fbeta(f_thresh, 2) for f_thresh in thresholds], axis=0) / np.linalg.norm(thresholds)
#
#
# def iou_new(preds, trues, threshold):
#     beta = 2
#     beta_square = beta ** 2
#
#     def iou_inner(pred, true):
#         if pred.sum() == true.sum():
#             return 1
#
#         tp = np.bitwise_and(pred, true).sum()  # m11
#         fp = np.bitwise_and(pred, 1 - true).sum()  # m01
#         fn = np.bitwise_and(1 - pred, true).sum()  # m10
#         # return m11 / (m01, m10 + m11 + eps)
#         return (1 + beta_square) / ((1 + beta_square) * tp + beta_square * fn + fp + eps)
#
#     preds_inner = np.squeeze(preds.data.cpu().numpy().copy(), axis=1)
#     trues_inner = np.squeeze(trues.data.cpu().numpy().copy(), axis=1)
#     preds_inner = np.reshape(preds_inner, (preds_inner.shape[0], preds_inner.shape[1] * preds_inner.shape[2]))
#     trues_inner = np.reshape(trues_inner, (trues_inner.shape[0], trues_inner.shape[1] * trues_inner.shape[2]))
#     preds_inner = np.clip(preds_inner, 0, 1)
#
#     preds_inner[preds_inner < threshold] = 0
#     preds_inner[preds_inner >= threshold] = 1
#
#     preds_inner = preds_inner.astype(np.int)
#     trues_inner = trues_inner.astype(np.int)
#
#     iou_values = np.array([iou_inner(true, pred) for pred, true in zip(preds_inner, trues_inner)])
#     return np.mean([iou_values > thresh for thresh in np.linspace(0.5, 0.95, 10)], axis=0)
#
#
# def IoU(mask1, mask2):
#     Inter = np.sum((mask1 > 0) & (mask2 > 0))
#     Union = np.sum((mask1 > 0) | (mask2 > 0))
#
#     return Inter / (1e-8 + Union)
#
#
# def fscore(tp, fn, fp, beta=2.):
#     if tp + fn + fp < 1:
#         return 1.
#     num = (1 + beta ** 2) * tp
#     return num / (num + beta ** 2 * fn + fp)
#
#
# def confusion_counts(predict_mask_seq, truth_mask_seq, iou_thresh=0.5):
#     if len(predict_mask_seq) + len(truth_mask_seq) == 0:
#         tp, fn, fp = 0, 0, 0
#         return tp, fn, fp
#
#     if len(predict_mask_seq) == 1 and len(truth_mask_seq) == 1 and np.sum(predict_mask_seq) == 0 and np.sum(truth_mask_seq) == 0:
#         return 1, 0, 0
#
#     pred_hits = np.zeros(len(predict_mask_seq), dtype=np.bool)  # 0 miss, 1 hit
#     truth_hits = np.zeros(len(truth_mask_seq), dtype=np.bool)  # 0 miss, 1 hit
#
#     for p, pred_mask in enumerate(predict_mask_seq):
#         for t, truth_mask in enumerate(truth_mask_seq):
#             if IoU(pred_mask, truth_mask) > iou_thresh:
#                 truth_hits[t] = True
#                 pred_hits[p] = True
#
#     tp = np.sum(pred_hits)
#     fn = len(truth_mask_seq) - np.sum(truth_hits)
#     fp = len(predict_mask_seq) - tp
#
#     return tp, fn, fp
#
#
# def mean_fscore(preds, trues, threshold, iou_thresholds=None, beta=2.):
#     def clip_to_the_masks(pred, true):
#         _, pred_cntrs, _ = cv2.findContours((np.clip(pred, 0, 1) * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#         pred_masks = [cv2.drawContours(np.zeros_like(pred, dtype=np.uint8), [c], 0, 1, -1).astype(np.float32) for c in pred_cntrs]
#         _, true_cntrs, _ = cv2.findContours((np.clip(true, 0, 1) * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#         true_masks = [cv2.drawContours(np.zeros_like(true, dtype=np.uint8), [c], 0, 1, -1).astype(np.float32) for c in true_cntrs]
#
#         if len(pred_masks) == 0:
#             pred_masks = [np.zeros_like(pred).astype(np.float32)]
#         if len(true_masks) == 0:
#             true_masks = [np.zeros_like(true).astype(np.float32)]
#         return pred_masks, true_masks
#
#     """ calculates the average FScore for the predictions in an image over
#     the iou_thresholds sets.
#     predict_mask_seq: list of masks of the predicted objects in the image
#     truth_mask_seq: list of masks of ground-truth objects in the image
#     """
#
#     if iou_thresholds is None:
#         iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
#
#     preds_inner = np.squeeze(preds.data.cpu().numpy().copy(), axis=1)
#     trues_inner = np.squeeze(trues.data.cpu().numpy().copy(), axis=1)
#     # preds_inner = np.reshape(preds_inner, (preds_inner.shape[0], preds_inner.shape[1] * preds_inner.shape[2]))
#     # trues_inner = np.reshape(trues_inner, (trues_inner.shape[0], trues_inner.shape[1] * trues_inner.shape[2]))
#     # preds_inner = np.clip(preds_inner, 0, 1)
#
#     preds_inner[preds_inner < threshold] = 0
#     preds_inner[preds_inner >= threshold] = 1
#
#     # preds_inner = preds_inner.astype(np.int)
#     # trues_inner = trues_inner.astype(np.int)
#
#     masks_per_image = [clip_to_the_masks(preds_inner[i], trues_inner[i]) for i in range(preds_inner.shape[0])]
#
#     return [np.mean([fscore(tp, fn, fp, beta) for (tp, fn, fp) in [confusion_counts(masks[0], masks[1], iou_thresh) for iou_thresh in iou_thresholds]]) for masks in masks_per_image]

def IoU(pred, targs):
    intersection = (pred * targs).sum()
    return intersection / ((pred + targs).sum() - intersection + 1.0)


def fbeta(preds, trues, threshold=0):
    n_th = 10
    b = 4
    thresholds = [0.5 + 0.05 * i for i in range(n_th)]

    preds_inner = np.squeeze(preds.data.cpu().numpy().copy(), axis=1)
    trues_inner = np.squeeze(trues.data.cpu().numpy().copy(), axis=1)
    preds_inner[preds_inner < threshold] = 0
    preds_inner[preds_inner >= threshold] = 1

    n_masks = len(trues_inner)
    n_pred = len(preds_inner)
    ious = []
    for mask in trues_inner:
        buf = []
        for p in preds_inner:
            buf.append(IoU(p, mask))
        ious.append(buf)
    res = []
    for t in thresholds:
        tp, fp, fn = 0, 0, 0
        for i in range(n_masks):
            match = False
            for j in range(n_pred):
                if ious[i][j] > t:
                    match = True
            if not match:
                fn += 1

        for j in range(n_pred):
            match = False
            for i in range(n_masks):
                if ious[i][j] > t:
                    match = True
            if match:
                tp += 1
            else:
                fp += 1
        res.append(((b + 1) * tp) / ((b + 1) * tp + b * fn + fp))
    return np.array(res)
