import torch

from piepline.train_config import MetricsProcessor, TrainStage
from piepline.train_config.train_config import ValidationStage, TrainConfig
from piepline.utils.fsm import FileStructManager
from piepline import Predictor, Trainer
from tests.common import UseFileStructure

from tests.data_processor_test import SimpleModel, SimpleLoss
from tests.data_producer_test import TestDataProducer


class PredictTest(UseFileStructure):
    def test_predict(self):
        model = SimpleModel()
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)

        metrics_processor = MetricsProcessor()
        stages = [TrainStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]),
                             metrics_processor),
                  ValidationStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]),
                                  metrics_processor)]
        Trainer(TrainConfig(model, stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=1)), fsm)\
            .set_epoch_num(1).train()

        fsm = FileStructManager(base_dir=self.base_dir, is_continue=True)
        Predictor(model, fsm).predict({'data': torch.rand(1, 3)})
