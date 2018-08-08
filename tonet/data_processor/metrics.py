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


def iou(preds, trues, threshold=None):
    def fbeta(threshold_shift=0, beta=2):
        y_pred_bin = np.round(preds_inner + threshold_shift)

        tp = np.sum(np.round(trues_inner * y_pred_bin), axis=1) + eps
        fp = np.sum(np.round(np.clip(y_pred_bin - trues_inner, 0, 1)), axis=1)
        fn = np.sum(np.round(np.clip(trues_inner - preds_inner, 0, 1)), axis=1)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        beta_squared = beta ** 2
        return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall)

    preds_inner = np.squeeze(preds.data.cpu().numpy().copy(), axis=1)
    trues_inner = np.squeeze(trues.data.cpu().numpy().copy(), axis=1)
    preds_inner = np.reshape(preds_inner, (preds_inner.shape[0], preds_inner.shape[1] * preds_inner.shape[2]))
    trues_inner = np.reshape(trues_inner, (trues_inner.shape[0], trues_inner.shape[1] * trues_inner.shape[2]))
    preds_inner = np.clip(preds_inner, 0, 1)

    if threshold is not None:
        preds_inner[preds_inner < threshold] = 0
        preds_inner[preds_inner >= threshold] = 1

    thresholds = np.arange(0.5, 0.96, 0.05)
    return np.sum([fbeta(f_thresh, 2) for f_thresh in thresholds], axis=0) / np.linalg.norm(thresholds)


def iou_new(preds, trues, threshold):
    beta = 2
    beta_square = beta ** 2

    def iou_inner(pred, true):
        if pred.sum() == true.sum():
            return 1

        tp = np.bitwise_and(pred, true).sum()  # m11
        fp = np.bitwise_and(pred, 1 - true).sum()  # m01
        fn = np.bitwise_and(1 - pred, true).sum()  # m10
        # return m11 / (m01, m10 + m11 + eps)
        return (1 + beta_square) / ((1 + beta_square) * tp + beta_square * fn + fp + eps)

    preds_inner = np.squeeze(preds.data.cpu().numpy().copy(), axis=1)
    trues_inner = np.squeeze(trues.data.cpu().numpy().copy(), axis=1)
    preds_inner = np.reshape(preds_inner, (preds_inner.shape[0], preds_inner.shape[1] * preds_inner.shape[2]))
    trues_inner = np.reshape(trues_inner, (trues_inner.shape[0], trues_inner.shape[1] * trues_inner.shape[2]))
    preds_inner = np.clip(preds_inner, 0, 1)

    preds_inner[preds_inner < threshold] = 0
    preds_inner[preds_inner >= threshold] = 1

    preds_inner = preds_inner.astype(np.int)
    trues_inner = trues_inner.astype(np.int)

    iou_values = np.array([iou_inner(true, pred) for pred, true in zip(preds_inner, trues_inner)])
    return np.mean([iou_values > thresh for thresh in np.linspace(0.5, 0.95, 10)], axis=0)
