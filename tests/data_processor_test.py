import shutil
import unittest

import torch
import numpy as np

from neural_pipeline.data_processor import DataProcessor
from neural_pipeline.data_processor.state_manager import StateManager
from neural_pipeline.utils.file_structure_manager import FileStructManager
from neural_pipeline.utils.utils import dict_pair_recursive_bypass

__all__ = ['DataProcessorTest']


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)


class NonStandartIOModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 1)

    def forward(self, x):
        res1 = self.fc(x['data1'])
        res2 = self.fc(x['data2'])
        return {'res1': res1, 'res2': res2}


class DataProcessorTest(unittest.TestCase):
    def test_initialisation(self):
        fsm = FileStructManager(checkpoint_dir_path='checkpoints', logdir_path='data', prefix=None)
        # state_manager = StateManager(fsm)
        model = SimpleModel()
        DataProcessor(model=model, file_struct_manager=fsm, is_cuda=False)

    def test_prediction_output(self):
        fsm = FileStructManager(checkpoint_dir_path='checkpoints', logdir_path='data', prefix=None)
        model = SimpleModel()
        dp = DataProcessor(model=model, file_struct_manager=fsm, is_cuda=False)
        self.assertFalse(model.fc.weight.is_cuda)
        res = dp.predict({'data': torch.rand(1, 3)})
        self.assertIs(type(res), torch.Tensor)

        model = NonStandartIOModel()
        dp = DataProcessor(model=model, file_struct_manager=fsm, is_cuda=False)
        self.assertFalse(model.fc.weight.is_cuda)
        res = dp.predict({'data': {'data1': torch.rand(1, 3), 'data2': torch.rand(1, 3)}})
        self.assertIs(type(res), dict)
        self.assertIn('res1', res)
        self.assertIs(type(res['res1']), torch.Tensor)
        self.assertIn('res2', res)
        self.assertIs(type(res['res2']), torch.Tensor)

    def test_predict(self):
        fsm = FileStructManager(checkpoint_dir_path='checkpoints', logdir_path='data', prefix=None)
        model = SimpleModel().train()
        dp = DataProcessor(model=model, file_struct_manager=fsm, is_cuda=False)
        self.assertFalse(model.fc.weight.is_cuda)
        self.assertTrue(model.training)
        res = dp.predict({'data': torch.rand(1, 3)})
        self.assertFalse(model.training)
        self.assertFalse(res.requires_grad)
        self.assertIsNone(res.grad)

    def test_continue_from_checkpoint(self):
        def on_node(n1, n2):
            self.assertTrue(np.array_equal(n1.numpy(), n2.numpy()))
        fsm = FileStructManager(checkpoint_dir_path='checkpoints', logdir_path='data', prefix=None)
        model = SimpleModel().train()
        dp = DataProcessor(model=model, file_struct_manager=fsm, is_cuda=False)
        before_state_dict = model.state_dict().copy()
        dp._model.save_weights()
        dp.load()
        after_state_dict = model.state_dict().copy()

        dict_pair_recursive_bypass(before_state_dict, after_state_dict, on_node)

    def tearDown(self):
        shutil.rmtree('checkpoints')
        shutil.rmtree('data')
