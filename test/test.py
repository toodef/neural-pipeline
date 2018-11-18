import torch
import unittest

from neural_pipeline.tonet.data_processor import metrics
import numpy as np


class TestMetrics(unittest.TestCase):
    def test_jaccard(self):
        preds = np.zeros((1, 1, 2, 2))
        trues = np.zeros((1, 1, 2, 2))

        preds[0, 0, 0, 0] = 1
        trues[0, 0, 0, 0] = 1

        self.assertEqual(metrics.jaccard(torch.autograd.Variable(torch.from_numpy(preds)), torch.autograd.Variable(torch.from_numpy(trues))), 1)

        preds[0, 0, 0, 0] = 1
        trues[0, 0, 0, 0] = 0

        self.assertAlmostEqual(metrics.jaccard(torch.autograd.Variable(torch.from_numpy(preds)), torch.autograd.Variable(torch.from_numpy(trues))), 0, delta=metrics.eps)

        preds[0, 0, 0, 0] = 0.5
        trues[0, 0, 0, 0] = 1

        self.assertAlmostEqual(metrics.jaccard(torch.autograd.Variable(torch.from_numpy(preds)), torch.autograd.Variable(torch.from_numpy(trues))), 0.5, delta=metrics.eps)

        preds[0, 0, 0, 0] = 0.5
        trues[0, 0, 0, 0] = 1

        self.assertAlmostEqual(metrics.masked_jaccard(torch.autograd.Variable(torch.from_numpy(preds)), torch.autograd.Variable(torch.from_numpy(trues)), threshold=0.5), 1, delta=metrics.eps)
        self.assertAlmostEqual(metrics.masked_jaccard(torch.autograd.Variable(torch.from_numpy(preds)), torch.autograd.Variable(torch.from_numpy(trues)), threshold=0.49), 1, delta=metrics.eps)
        self.assertAlmostEqual(metrics.masked_jaccard(torch.autograd.Variable(torch.from_numpy(preds)), torch.autograd.Variable(torch.from_numpy(trues)), threshold=0.6), 0, delta=metrics.eps)
