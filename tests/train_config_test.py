import unittest

import numpy as np
import torch
from torch.nn import functional as F
from torch import Tensor

from neural_pipeline import Trainer
from neural_pipeline.data_producer import DataProducer
from neural_pipeline.train_config.train_config import MetricsGroup, AbstractMetric, TrainStage, MetricsProcessor, TrainConfig
from neural_pipeline.utils.file_structure_manager import FileStructManager
from tests.common import UseResources
from tests.data_processor_test import SimpleModel, SimpleLoss

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


class TrainConfigTest(unittest.TestCase, UseResources):
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

        for m in metrics_group_lv1.metrics():
            for v1, v2 in zip(values, m.get_values()):
                self.assertAlmostEqual(v1, v2, delta=1e-5)
        for m in metrics_group_lv2.metrics():
            for v1, v2 in zip(values, m.get_values()):
                self.assertAlmostEqual(v1, v2, delta=1e-5)

        metrics_group_lv1.reset()
        self.assertEqual(metrics_group_lv1.metrics()[0].get_values().size, 0)
        self.assertEqual(metrics_group_lv2.metrics()[0].get_values().size, 0)

    def test_train_stage(self):
        data_producer = DataProducer([[{'data': torch.rand(1, 3), 'target': torch.rand(1)} for _ in list(range(20))]])
        metrics_processor = FakeMetricsProcessor()
        train_stage = TrainStage(data_producer, metrics_processor).enable_hard_negative_mining(0.1)

        fsm = FileStructManager(checkpoint_dir_path=self.checkpoints_dir, logdir_path=self.logdir, prefix=None)
        model = SimpleModel()
        Trainer(model, TrainConfig([train_stage], SimpleLoss(), torch.optim.SGD(model.parameters(), lr=1), 'exp'), fsm, is_cuda=False) \
            .set_epoch_num(1).train()

        self.assertEqual(metrics_processor.call_num, len(data_producer))


if __name__ == '__main__':
    unittest.main()
