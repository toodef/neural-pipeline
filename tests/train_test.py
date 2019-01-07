import os
import shutil
import unittest

import torch

from neural_pipeline import Trainer
from neural_pipeline.data_processor import Model
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
                          TrainConfig([], torch.nn.L1Loss(), torch.optim.SGD(model.model().parameters(), lr=1), 'exp'),
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
        Trainer(model, TrainConfig(stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=1), 'exp'), fsm, is_cuda=False)\
            .set_epoch_num(1).train()

    def tearDown(self):
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir, ignore_errors=True)
        if os.path.exists(self.checkpoints_dir):
            shutil.rmtree(self.checkpoints_dir, ignore_errors=True)
