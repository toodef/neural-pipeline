import unittest

import numpy as np
import torch
from torch.nn import functional as F
from torch import Tensor

from piepline import Trainer
from piepline.data_producer import DataProducer
from piepline.train_config.train_config import MetricsGroup, AbstractMetric, TrainStage, MetricsProcessor, TrainConfig
from piepline.utils.fsm import FileStructManager
from tests.common import UseFileStructure
from tests.data_processor_test import SimpleModel, SimpleLoss
from tests.data_producer_test import TestDataProducer

__all__ = ['TrainConfigTest']


class SimpleMetric(AbstractMetric):
    def __init__(self):
        super().__init__('SimpleMetric')

    def calc(self, output: Tensor, target: Tensor) -> np.ndarray or float:
        return F.pairwise_distance(output, target, p=2).numpy()


class FakeMetricsProcessor(MetricsProcessor):
    def __init__(self):
        super().__init__()
        self.call_num = 0

    def calc_metrics(self, output, target):
        self.call_num += 1


class TrainConfigTest(UseFileStructure):
    def test_metric(self):
        metric = SimpleMetric()

        for i in range(10):
            output, target = torch.rand(1, 3), torch.rand(1, 3)
            res = metric.calc(output, target)[0]
            self.assertAlmostEqual(res, np.linalg.norm(output.numpy() - target.numpy()), delta=1e-5)

        vals = metric.get_values()
        self.assertEqual(vals.size, 0)

        values = []
        for i in range(10):
            output, target = torch.rand(1, 3), torch.rand(1, 3)
            metric._calc(output, target)
            values.append(np.linalg.norm(output.numpy() - target.numpy()))

        vals = metric.get_values()
        self.assertEqual(vals.size, len(values))
        for v1, v2 in zip(values, vals):
            self.assertAlmostEqual(v1, v2, delta=1e-5)

        metric.reset()
        self.assertEqual(metric.get_values().size, 0)

        self.assertEqual(metric.name(), "SimpleMetric")

    def test_metrics_group_nested(self):
        metrics_group_lv1 = MetricsGroup('lvl')
        metrics_group_lv2 = MetricsGroup('lv2')
        metrics_group_lv1.add(metrics_group_lv2)
        self.assertTrue(metrics_group_lv1.have_groups())
        self.assertRaises(MetricsGroup.MGException, lambda: metrics_group_lv2.add(MetricsGroup('lv3')))

        metrics_group_lv1 = MetricsGroup('lvl')
        metrics_group_lv2 = MetricsGroup('lv2')
        metrics_group_lv3 = MetricsGroup('lv2')
        metrics_group_lv2.add(metrics_group_lv3)
        self.assertRaises(MetricsGroup.MGException, lambda: metrics_group_lv1.add(metrics_group_lv2))

    def test_metrics_group_calculation(self):
        metrics_group_lv1 = MetricsGroup('lvl').add(SimpleMetric())
        metrics_group_lv2 = MetricsGroup('lv2').add(SimpleMetric())
        metrics_group_lv1.add(metrics_group_lv2)

        values = []
        for i in range(10):
            output, target = torch.rand(1, 3), torch.rand(1, 3)
            metrics_group_lv1.calc(output, target)
            values.append(np.linalg.norm(output.numpy() - target.numpy()))

        for metrics_group in [metrics_group_lv1, metrics_group_lv2]:
            for m in metrics_group.metrics():
                for v1, v2 in zip(values, m.get_values()):
                    self.assertAlmostEqual(v1, v2, delta=1e-5)

        metrics_group_lv1.reset()
        self.assertEqual(metrics_group_lv1.metrics()[0].get_values().size, 0)
        self.assertEqual(metrics_group_lv2.metrics()[0].get_values().size, 0)

    def test_metrics_pocessor_calculation(self):
        metrics_group_lv11 = MetricsGroup('lvl').add(SimpleMetric())
        metrics_group_lv21 = MetricsGroup('lv2').add(SimpleMetric())
        metrics_group_lv11.add(metrics_group_lv21)
        metrics_processor = MetricsProcessor()
        metrics_group_lv12 = MetricsGroup('lvl').add(SimpleMetric())
        metrics_group_lv22 = MetricsGroup('lv2').add(SimpleMetric())
        metrics_group_lv12.add(metrics_group_lv22)
        metrics_processor.add_metrics_group(metrics_group_lv11)
        metrics_processor.add_metrics_group(metrics_group_lv12)
        m1, m2 = SimpleMetric(), SimpleMetric()
        metrics_processor.add_metric(m1)
        metrics_processor.add_metric(m2)

        values = []
        for i in range(10):
            output, target = torch.rand(1, 3), torch.rand(1, 3)
            metrics_processor.calc_metrics(output, target)
            values.append(np.linalg.norm(output.numpy() - target.numpy()))

        for metrics_group in [metrics_group_lv11, metrics_group_lv21, metrics_group_lv12, metrics_group_lv22]:
            for m in metrics_group.metrics():
                for v1, v2 in zip(values, m.get_values()):
                    self.assertAlmostEqual(v1, v2, delta=1e-5)
        for m in [m1, m2]:
            for v1, v2 in zip(values, m.get_values()):
                self.assertAlmostEqual(v1, v2, delta=1e-5)

        metrics_processor.reset_metrics()
        self.assertEqual(metrics_group_lv11.metrics()[0].get_values().size, 0)
        self.assertEqual(metrics_group_lv21.metrics()[0].get_values().size, 0)
        self.assertEqual(metrics_group_lv12.metrics()[0].get_values().size, 0)
        self.assertEqual(metrics_group_lv22.metrics()[0].get_values().size, 0)
        self.assertEqual(m1.get_values().size, 0)
        self.assertEqual(m2.get_values().size, 0)

    def test_metrics_and_groups_collection(self):
        m1 = SimpleMetric()
        name = 'lv1'
        metrics_group_lv1 = MetricsGroup(name)
        self.assertEqual(metrics_group_lv1.metrics(), [])
        metrics_group_lv1.add(m1)
        self.assertEqual(metrics_group_lv1.groups(), [])
        self.assertEqual(metrics_group_lv1.metrics(), [m1])

        metrics_group_lv2 = MetricsGroup('lv2').add(SimpleMetric())
        metrics_group_lv1.add(metrics_group_lv2)
        self.assertEqual(metrics_group_lv1.groups(), [metrics_group_lv2])
        self.assertEqual(metrics_group_lv1.metrics(), [m1])

        metrics_group_lv22 = MetricsGroup('lv2').add(SimpleMetric())
        metrics_group_lv1.add(metrics_group_lv22)
        self.assertEqual(metrics_group_lv1.groups(), [metrics_group_lv2, metrics_group_lv22])
        self.assertEqual(metrics_group_lv1.metrics(), [m1])

        self.assertEqual(metrics_group_lv1.name(), name)

    def test_train_stage(self):
        data_producer = DataProducer([{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))])
        metrics_processor = FakeMetricsProcessor()
        train_stage = TrainStage(data_producer, metrics_processor).enable_hard_negative_mining(0.1)

        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        model = SimpleModel()
        Trainer(TrainConfig(model, [train_stage], SimpleLoss(), torch.optim.SGD(model.parameters(), lr=1)), fsm) \
            .set_epoch_num(1).train()

        self.assertEqual(metrics_processor.call_num, len(data_producer))

    def test_hard_negatives_mining(self):
        with self.assertRaises(ValueError):
            stage = TrainStage(None).enable_hard_negative_mining(0)
        with self.assertRaises(ValueError):
            stage = TrainStage(None).enable_hard_negative_mining(1)
        with self.assertRaises(ValueError):
            stage = TrainStage(None).enable_hard_negative_mining(-1)
        with self.assertRaises(ValueError):
            stage = TrainStage(None).enable_hard_negative_mining(1.1)

        dp = TestDataProducer([{'data': torch.Tensor([i]), 'target': torch.rand(1)} for i in list(range(20))]).pass_indices(True)
        stage = TrainStage(dp).enable_hard_negative_mining(0.1)
        losses = np.random.rand(20)
        samples = []

        def on_batch(batch, data_processor):
            samples.append(batch)
            stage.hnm._losses = np.array([0])

        stage.hnm._process_batch = on_batch
        stage.hnm.exec(None, losses, [[str(i)] for i in range(20)])

        self.assertEqual(len(samples), 2)

        losses = [float(v) for v in losses]
        idxs = [int(s['data']) for s in samples]
        max_losses = [losses[i] for i in idxs]
        idxs.sort(reverse=True)
        for i in idxs:
            del losses[i]

        for l in losses:
            self.assertLess(l, min(max_losses))

        stage.on_epoch_end()

        self.assertIsNone(stage.hnm._losses)

        stage.disable_hard_negative_mining()
        self.assertIsNone(stage.hnm)

        for data in dp:
            self.assertIn('data_idx', data)


if __name__ == '__main__':
    unittest.main()
