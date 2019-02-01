import os
import shutil
import unittest
from random import randint

import torch

from neural_pipeline import Trainer
from neural_pipeline.data_processor import Model
from neural_pipeline.train import DecayingLR
from neural_pipeline.train_config import TrainConfig, TrainStage, MetricsProcessor
from neural_pipeline.train_config.train_config import ValidationStage
from neural_pipeline.utils.file_structure_manager import FileStructManager
from tests.data_processor_test import SimpleModel
from tests.data_producer_test import TestDataProducer

__all__ = ['TrainTest']


class SimpleLoss(torch.nn.Module):
    def forward(self, output, target):
        return output / target


class TrainTest(unittest.TestCase):
    logdir = 'logs'
    checkpoints_dir = 'tensorboard_logs'

    def test_base_ops(self):
        fsm = FileStructManager(checkpoint_dir_path=self.checkpoints_dir, logdir_path=self.logdir, prefix=None)
        model = Model(SimpleModel(), fsm)

        trainer = Trainer(model,
                          TrainConfig([], torch.nn.L1Loss(), torch.optim.SGD(model.model().parameters(), lr=1)),
                          fsm, is_cuda=False)
        with self.assertRaises(Trainer.TrainerException):
            trainer.train()

    def test_train(self):
        fsm = FileStructManager(checkpoint_dir_path=self.checkpoints_dir, logdir_path=self.logdir, prefix=None)
        model = SimpleModel()
        metrics_processor = MetricsProcessor()
        stages = [TrainStage(TestDataProducer([[{'data': torch.rand(1, 3), 'target': torch.rand(1)}
                                                for _ in list(range(20))]]), metrics_processor),
                  ValidationStage(TestDataProducer([[{'data': torch.rand(1, 3), 'target': torch.rand(1)}
                                                     for _ in list(range(20))]]), metrics_processor)]
        Trainer(model, TrainConfig(stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=1)), fsm, is_cuda=False) \
            .set_epoch_num(1).train()

    def test_decaynig_lr(self):
        step_num = 0

        def target_value_clbk() -> float:
            return 1 / step_num

        lr = DecayingLR(0.1, 0.5, 3, target_value_clbk)
        old_val = None
        for i in range(1, 30):
            step_num = i
            value = lr.value()
            if old_val is None:
                old_val = value
                continue

            self.assertAlmostEqual(value, old_val, delta=1e-6)
            old_val = value

        step_num = 0

        def target_value_clbk() -> float:
            return 1

        lr = DecayingLR(0.1, 0.5, 3, target_value_clbk)
        old_val = None
        for i in range(1, 30):
            step_num = i
            value = lr.value()
            if old_val is None:
                old_val = value
                continue

            if i % 3 == 0:
                self.assertAlmostEqual(value, old_val * 0.5, delta=1e-6)
            old_val = value

        val = randint(1, 1000)
        lr.set_value(val)
        self.assertEqual(val, lr.value())

    def test_lr_decaying(self):
        fsm = FileStructManager(checkpoint_dir_path=self.checkpoints_dir, logdir_path=self.logdir, prefix=None)
        model = SimpleModel()
        metrics_processor = MetricsProcessor()
        stages = [TrainStage(TestDataProducer([[{'data': torch.rand(1, 3), 'target': torch.rand(1)}
                                                for _ in list(range(20))]]), metrics_processor),
                  ValidationStage(TestDataProducer([[{'data': torch.rand(1, 3), 'target': torch.rand(1)}
                                                     for _ in list(range(20))]]), metrics_processor)]
        trainer = Trainer(model, TrainConfig(stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=0.1)),
                          fsm, is_cuda=False).set_epoch_num(10)

        def target_value_clbk() -> float:
            return 1

        trainer.enable_lr_decaying(0.5, 3, target_value_clbk)
        trainer.train()

        self.assertAlmostEqual(trainer.data_processor().get_lr(), 0.1 * (0.5 ** 3), delta=1e-6)

    def tearDown(self):
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir, ignore_errors=True)
        if os.path.exists(self.checkpoints_dir):
            shutil.rmtree(self.checkpoints_dir, ignore_errors=True)
