import unittest

import torch

from neural_pipeline.train_config import MetricsProcessor, TrainStage
from neural_pipeline.train_config.train_config import ValidationStage, TrainConfig
from neural_pipeline.utils.file_structure_manager import FileStructManager
from neural_pipeline import Predictor, Trainer

from tests.data_processor_test import SimpleModel, SimpleLoss
from tests.data_producer_test import TestDataProducer


class TrainTest(unittest.TestCase):
    logdir = 'logs'
    checkpoints_dir = 'tensorboard_logs'

    def test_predict(self):
        model = SimpleModel()
        fsm = FileStructManager(checkpoint_dir_path=self.checkpoints_dir, logdir_path=self.logdir, prefix=None)

        metrics_processor = MetricsProcessor()
        stages = [TrainStage(TestDataProducer([[{'data': torch.rand(1, 3), 'target': torch.rand(1)}
                                                for _ in list(range(20))]]), metrics_processor),
                  ValidationStage(TestDataProducer([[{'data': torch.rand(1, 3), 'target': torch.rand(1)}
                                                     for _ in list(range(20))]]), metrics_processor)]
        Trainer(model, TrainConfig(stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=1), 'exp'), fsm, is_cuda=False)\
            .set_epoch_num(1).train()

        Predictor(model, fsm).predict({'data': torch.rand(1, 3)})
