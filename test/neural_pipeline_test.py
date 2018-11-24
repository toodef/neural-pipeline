import unittest

from tonet.neural_pipeline.train_pipeline.train_pipeline import MetricsGroup


class NeuralPipelineTest(unittest.TestCase):
    def test_metrics(self):
        metrics_group_lv1 = MetricsGroup('lvl')
        metrics_group_lv2 = MetricsGroup('lv2')
        metrics_group_lv1.add(metrics_group_lv2)
        self.assertRaises(metrics_group_lv2.add(MetricsGroup('lv3')))


if __name__ == '__main__':
    unittest.main()
