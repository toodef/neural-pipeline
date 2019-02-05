from random import randint

import torch
import numpy as np

from neural_pipeline import Trainer
from neural_pipeline.train import DecayingLR
from neural_pipeline.train_config import TrainConfig, TrainStage, MetricsProcessor
from neural_pipeline.train_config.train_config import ValidationStage
from neural_pipeline.utils.file_structure_manager import FileStructManager
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

        trainer = Trainer(model,
                          TrainConfig([], torch.nn.L1Loss(), torch.optim.SGD(model.parameters(), lr=1)),
                          fsm)
        with self.assertRaises(Trainer.TrainerException):
            trainer.train()

    def test_train(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()
        metrics_processor = MetricsProcessor()
        stages = [TrainStage(TestDataProducer([[{'data': torch.rand(1, 3), 'target': torch.rand(1)}
                                                for _ in list(range(20))]]), metrics_processor),
                  ValidationStage(TestDataProducer([[{'data': torch.rand(1, 3), 'target': torch.rand(1)}
                                                     for _ in list(range(20))]]), metrics_processor)]
        Trainer(model, TrainConfig(stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=1)), fsm) \
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
        stages = [TrainStage(TestDataProducer([[{'data': torch.rand(1, 3), 'target': torch.rand(1)}
                                                for _ in list(range(20))]]), metrics_processor),
                  ValidationStage(TestDataProducer([[{'data': torch.rand(1, 3), 'target': torch.rand(1)}
                                                     for _ in list(range(20))]]), metrics_processor)]
        trainer = Trainer(model, TrainConfig(stages, SimpleLoss(), torch.optim.SGD(model.parameters(), lr=0.1)),
                          fsm).set_epoch_num(10)

        def target_value_clbk() -> float:
            return 1

        trainer.enable_lr_decaying(0.5, 3, target_value_clbk)
        trainer.train()

        self.assertAlmostEqual(trainer.data_processor().get_lr(), 0.1 * (0.5 ** 3), delta=1e-6)

    def test_hard_negatives_mining(self):
        with self.assertRaises(ValueError):
            stage = TrainStage(None).enable_hard_negative_mining(0)
        with self.assertRaises(ValueError):
            stage = TrainStage(None).enable_hard_negative_mining(1)
        with self.assertRaises(ValueError):
            stage = TrainStage(None).enable_hard_negative_mining(-1)
        with self.assertRaises(ValueError):
            stage = TrainStage(None).enable_hard_negative_mining(1.1)

        stage = TrainStage(TestDataProducer([[{'data': torch.Tensor([i]), 'target': torch.rand(1)}
                                              for i in list(range(20))]])
                           ).enable_hard_negative_mining(0.1)
        losses = np.random.rand(20)

        samples = []

        def on_batch(batch, data_processor):
            samples.append(batch)
            stage.hnm._losses = np.array([0])

        stage.hnm._process_batch = on_batch
        stage.hnm.exec(None, losses, [['0_{}'.format(i)] for i in range(20)])

        self.assertEqual(len(samples), 2)

        losses = [float(v) for v in losses]

        def is_max(v):
            for val in losses[:len(losses) - 2]:
                self.assertLess(val, v)

        idxs = [int(s['data']) for s in samples]
        max_losses = [losses[i] for i in idxs]
        idxs.sort(reverse=True)
        for i in idxs:
            del losses[i]

        for l in losses:
            self.assertLess(l, min(max_losses))

        stage.on_epoch_end()

        self.assertIsNone(stage.hnm._losses)
