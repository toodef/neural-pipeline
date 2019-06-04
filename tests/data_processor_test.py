import os
import shutil
import unittest

import torch
import numpy as np

from neural_pipeline.data_processor import DataProcessor, TrainDataProcessor, Model
from neural_pipeline.utils import FileStructManager, dict_pair_recursive_bypass, CheckpointsManager
from neural_pipeline.train_config import TrainConfig
from tests.common import UseFileStructure, data_remove

__all__ = ['ModelTest', 'DataProcessorTest', 'TrainDataProcessorTest']


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)

    @staticmethod
    def dummy_input():
        return torch.rand(3)


def compare_two_models(unittest_obj: unittest.TestCase, model1: torch.nn.Module, model2: torch.nn.Module):
    def on_node(n1, n2):
        if n1.device == torch.device('cuda:0'):
            n1 = n1.to('cpu')
        if n2.device == torch.device('cuda:0'):
            n2 = n2.to('cpu')
        unittest_obj.assertTrue(np.array_equal(n1.numpy(), n2.numpy()))

    state_dict1 = model1.state_dict().copy()
    state_dict2 = model2.state_dict().copy()

    dict_pair_recursive_bypass(state_dict1, state_dict2, on_node)


class NonStandardIOModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 1)

    def forward(self, x):
        res1 = self.fc(x['data1'])
        res2 = self.fc(x['data2'])
        return {'res1': res1, 'res2': res2}


class ModelTest(UseFileStructure):
    def test_init(self):
        try:
            Model(SimpleModel())
        except:
            self.fail('Model initialization fail')

    def test_data_parallel(self):
        try:
            m = Model(torch.nn.DataParallel(SimpleModel()))
            m(SimpleModel.dummy_input())
        except:
            self.fail('Model DataParallel pass fail')

        if torch.cuda.is_available():
            try:
                device = torch.device('cuda')
                m = Model(torch.nn.DataParallel(SimpleModel())).to_device(device)
                m(SimpleModel.dummy_input().to(device))
            except:
                self.fail('Model DataParallel on gpu fail')

    def test_saving_and_loading(self):
        m = Model(SimpleModel())
        m.save_weights('weights.pth')
        m_new = Model(SimpleModel())
        m_new.load_weights('weights.pth')
        compare_two_models(self, m.model(), m_new.model())
        os.remove('weights.pth')

        m = SimpleModel()
        Model(torch.nn.DataParallel(m)).save_weights('weights.pth')
        m_new = Model(SimpleModel())
        m_new.load_weights('weights.pth')
        compare_two_models(self, m, m_new.model())
        os.remove('weights.pth')

        m = SimpleModel()
        Model(m).save_weights('weights.pth')
        m_new = SimpleModel()
        Model(torch.nn.DataParallel(m_new)).load_weights('weights.pth')
        compare_two_models(self, m, m_new)
        os.remove('weights.pth')

        m = SimpleModel()
        Model(torch.nn.DataParallel(m)).save_weights('weights.pth')
        m_new = SimpleModel()
        Model(torch.nn.DataParallel(m_new)).load_weights('weights.pth')
        compare_two_models(self, m, m_new)
        os.remove('weights.pth')


class DataProcessorTest(UseFileStructure):
    def test_initialisation(self):
        try:
            DataProcessor(model=SimpleModel())
        except:
            self.fail('DataProcessor initialisation raises exception')

    def test_prediction_output(self):
        model = SimpleModel()
        dp = DataProcessor(model=model)
        self.assertFalse(model.fc.weight.is_cuda)
        res = dp.predict({'data': torch.rand(1, 3)})
        self.assertIs(type(res), torch.Tensor)

        model = NonStandardIOModel()
        dp = DataProcessor(model=model)
        self.assertFalse(model.fc.weight.is_cuda)
        res = dp.predict({'data': {'data1': torch.rand(1, 3), 'data2': torch.rand(1, 3)}})
        self.assertIs(type(res), dict)
        self.assertIn('res1', res)
        self.assertIs(type(res['res1']), torch.Tensor)
        self.assertIn('res2', res)
        self.assertIs(type(res['res2']), torch.Tensor)

    def test_predict(self):
        model = SimpleModel().train()
        dp = DataProcessor(model=model)
        self.assertFalse(model.fc.weight.is_cuda)
        self.assertTrue(model.training)
        res = dp.predict({'data': torch.rand(1, 3)})
        self.assertFalse(model.training)
        self.assertFalse(res.requires_grad)
        self.assertIsNone(res.grad)

    @data_remove
    def test_continue_from_checkpoint(self):
        def on_node(n1, n2):
            self.assertTrue(np.array_equal(n1.numpy(), n2.numpy()))

        model = SimpleModel().train()
        dp = DataProcessor(model=model)
        with self.assertRaises(Model.ModelException):
            dp.save_state()
        try:
            fsm = FileStructManager(self.base_dir, is_continue=False)
            dp.set_checkpoints_manager(CheckpointsManager(fsm))
            dp.save_state()
        except:
            self.fail('Fail to DataProcessor load when CheckpointsManager was defined')

        del dp

        model_new = SimpleModel().train()
        dp = DataProcessor(model=model_new)

        with self.assertRaises(Model.ModelException):
            dp.load()
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=True)
        dp.set_checkpoints_manager(CheckpointsManager(fsm))
        try:
            fsm = FileStructManager(self.base_dir, is_continue=True)
            dp.set_checkpoints_manager(CheckpointsManager(fsm))
            dp.load()
        except:
            self.fail('Fail to DataProcessor load when CheckpointsManager was defined')

        compare_two_models(self, model, model_new)


class SimpleLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = torch.nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.res = None

    def forward(self, predict, target):
        self.res = self.module * (predict - target)
        return self.res


class TrainDataProcessorTest(UseFileStructure):
    def test_initialisation(self):
        model = SimpleModel()
        train_config = TrainConfig(model, [], torch.nn.Module(), torch.optim.SGD(model.parameters(), lr=0.1))
        try:
            TrainDataProcessor(train_config=train_config)
        except:
            self.fail('DataProcessor initialisation raises exception')

    def test_prediction_output(self):
        model = SimpleModel()
        train_config = TrainConfig(model, [], torch.nn.Module(), torch.optim.SGD(model.parameters(), lr=0.1))
        dp = TrainDataProcessor(train_config=train_config)
        self.assertFalse(model.fc.weight.is_cuda)
        res = dp.predict({'data': torch.rand(1, 3)}, is_train=False)
        self.assertIs(type(res), torch.Tensor)

        model = NonStandardIOModel()
        dp = TrainDataProcessor(train_config=train_config)
        self.assertFalse(model.fc.weight.is_cuda)
        res = dp.predict({'data': {'data1': torch.rand(1, 3), 'data2': torch.rand(1, 3)}}, is_train=False)
        self.assertIs(type(res), dict)
        self.assertIn('res1', res)
        self.assertIs(type(res['res1']), torch.Tensor)
        self.assertIn('res2', res)
        self.assertIs(type(res['res2']), torch.Tensor)

    def test_predict(self):
        model = SimpleModel().train()
        train_config = TrainConfig(model, [], torch.nn.Module(), torch.optim.SGD(model.parameters(), lr=0.1))
        dp = TrainDataProcessor(train_config=train_config)
        self.assertFalse(model.fc.weight.is_cuda)
        self.assertTrue(model.training)
        res = dp.predict({'data': torch.rand(1, 3)})
        self.assertFalse(model.training)
        self.assertFalse(res.requires_grad)
        self.assertIsNone(res.grad)

    def test_train(self):
        model = SimpleModel().train()
        train_config = TrainConfig(model, [], torch.nn.Module(), torch.optim.SGD(model.parameters(), lr=0.1))
        dp = TrainDataProcessor(train_config=train_config)

        self.assertFalse(model.fc.weight.is_cuda)
        self.assertTrue(model.training)
        res = dp.predict({'data': torch.rand(1, 3)}, is_train=True)
        self.assertTrue(model.training)
        self.assertTrue(res.requires_grad)
        self.assertIsNone(res.grad)

        with self.assertRaises(NotImplementedError):
            dp.process_batch({'data': torch.rand(1, 3), 'target': torch.rand(1)}, is_train=True)

        loss = SimpleLoss()
        train_config = TrainConfig(model, [], loss, torch.optim.SGD(model.parameters(), lr=0.1))
        dp = TrainDataProcessor(train_config=train_config)
        res = dp.process_batch({'data': torch.rand(1, 3), 'target': torch.rand(1)}, is_train=True)
        self.assertTrue(model.training)
        self.assertTrue(loss.module.requires_grad)
        self.assertIsNotNone(loss.module.grad)
        self.assertTrue(np.array_equal(res, loss.res.data.numpy()))

    @data_remove
    def test_continue_from_checkpoint(self):
        def on_node(n1, n2):
            self.assertTrue(np.array_equal(n1.numpy(), n2.numpy()))

        model = SimpleModel().train()
        loss = SimpleLoss()

        for optim in [torch.optim.SGD(model.parameters(), lr=0.1), torch.optim.Adam(model.parameters(), lr=0.1)]:
            train_config = TrainConfig(model, [], loss, optim)

            dp_before = TrainDataProcessor(train_config=train_config)
            before_state_dict = model.state_dict().copy()
            dp_before.update_lr(0.023)

            with self.assertRaises(Model.ModelException):
                dp_before.save_state()
            try:
                fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
                dp_before.set_checkpoints_manager(CheckpointsManager(fsm))
                dp_before.save_state()
            except:
                self.fail("Exception on saving state when 'CheckpointsManager' specified")

            fsm = FileStructManager(base_dir=self.base_dir, is_continue=True)
            dp_after = TrainDataProcessor(train_config=train_config)
            with self.assertRaises(Model.ModelException):
                dp_after.load()
            try:
                cm = CheckpointsManager(fsm)
                dp_after.set_checkpoints_manager(cm)
                dp_after.load()
            except:
                self.fail('DataProcessor initialisation raises exception')

            after_state_dict = model.state_dict().copy()

            dict_pair_recursive_bypass(before_state_dict, after_state_dict, on_node)
            self.assertEqual(dp_before.get_lr(), dp_after.get_lr())

            shutil.rmtree(self.base_dir)
