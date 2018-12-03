import torch
from torch import Tensor
import numpy as np
import unittest

from neural_pipeline.utils.utils import dict_recursive_bypass

__all__ = ['UtilsTest']


class UtilsTest(unittest.TestCase):
    def test_dict_recursive_bypass(self):
        d = {'data': np.array([1]), 'target': {'a': np.array([1]), 'b': np.array([1])}}
        d = dict_recursive_bypass(d, lambda k, v: torch.from_numpy(v))

        self.assertTrue(isinstance(d['data'], Tensor))
        self.assertTrue(isinstance(d['target']['a'], Tensor))
        self.assertTrue(isinstance(d['target']['b'], Tensor))


if __name__ == '__main__':
    unittest.main()
