import os
from random import randint

import torch
import numpy as np

from piepline import Trainer
from piepline.train import DecayingLR
from piepline.train_config import TrainConfig, TrainStage, MetricsProcessor
from piepline.train_config.train_config import ValidationStage
from piepline.utils.fsm import FileStructManager
from tests.common import UseFileStructure
from tests.data_processor_test import SimpleModel
from tests.data_producer_test import TestDataProducer

__all__ = ['TrainTest']


class SimpleLoss(torch.nn.Module):
    def forward(self, output, target):
        return output / target


class TrainTest(UseFileStructure):
    def test_base_ops(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()

        trainer = Trainer(TrainConfig(model, [], torch.nn.L1Loss(), torch.optim.SGD(model.parameters(), lr=1)),
                          fsm)
        with self.assertRaises(Trainer.TrainerException):
            trainer.train()

    def test_train(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()
        metrics_processor = MetricsProcessor()
        stages = [TrainStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]),
                             metrics_processor),
                  ValidationStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]),
                                  metrics_processor)]
        Trainer(TrainConfig(model, stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=1)), fsm) \
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
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()
        metrics_processor = MetricsProcessor()
        stages = [TrainStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]),
                             metrics_processor),
                  ValidationStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]),
                                  metrics_processor)]
        trainer = Trainer(TrainConfig(model, stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=0.1)),
                          fsm).set_epoch_num(10)

        def target_value_clbk() -> float:
            return 1

        trainer.enable_lr_decaying(0.5, 3, target_value_clbk)
        trainer.train()

        self.assertAlmostEqual(trainer.data_processor().get_lr(), 0.1 * (0.5 ** 3), delta=1e-6)

    def test_savig_states(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()
        metrics_processor = MetricsProcessor()
        stages = [TrainStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]),
                             metrics_processor)]
        trainer = Trainer(TrainConfig(model, stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=0.1)),
                          fsm).set_epoch_num(3)

        checkpoint_file = os.path.join(self.base_dir, 'checkpoints', 'last', 'last_checkpoint.zip')

        def on_epoch_end():
            self.assertTrue(os.path.exists(checkpoint_file))
            os.remove(checkpoint_file)

        trainer.add_on_epoch_end_callback(on_epoch_end)
        trainer.train()

    def test_savig_best_states(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()
        metrics_processor = MetricsProcessor()
        stages = [TrainStage(TestDataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]),
                             metrics_processor)]
        trainer = Trainer(TrainConfig(model, stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=0.1)),
                          fsm).set_epoch_num(3).enable_best_states_saving(lambda: np.mean(stages[0].get_losses()))

        checkpoint_file = os.path.join(self.base_dir, 'checkpoints', 'last', 'last_checkpoint.zip')
        best_checkpoint_file = os.path.join(self.base_dir, 'checkpoints', 'best', 'best_checkpoint.zip')

        class Val:
            def __init__(self):
                self.v = None

        first_val = Val()

        def on_epoch_end(val):
            if val.v is not None and np.mean(stages[0].get_losses()) < val.v:
                self.assertTrue(os.path.exists(best_checkpoint_file))
                os.remove(best_checkpoint_file)
                val.v = np.mean(stages[0].get_losses())
                return

            val.v = np.mean(stages[0].get_losses())

            self.assertTrue(os.path.exists(checkpoint_file))
            self.assertFalse(os.path.exists(best_checkpoint_file))
            os.remove(checkpoint_file)

        trainer.add_on_epoch_end_callback(lambda: on_epoch_end(first_val))
        trainer.train()
