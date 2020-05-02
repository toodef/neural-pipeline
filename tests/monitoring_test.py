import json
import os
import shutil

import torch
import numpy as np

from piepline.monitoring import LogMonitor
from piepline.train_config import MetricsGroup
from piepline.utils import FileStructManager
from tests.common import UseFileStructure, SimpleMetric

__all__ = ['MonitorLogTest']


class MonitorLogTest(UseFileStructure):
    def test_base_execution(self):
        fsm = FileStructManager(base_dir='data', is_continue=False)
        expected_out = os.path.join('data', 'monitors', 'metrics_log', 'metrics_log.json')
        try:
            with LogMonitor(fsm) as m:
                self.assertEqual(m._file, expected_out)
        except:
            self.fail('Fail initialisation')

        self.assertTrue(os.path.exists(expected_out) and os.path.isfile(expected_out))

    def metrics_processing(self, with_final_file: bool, final_file: str = None):
        fsm = FileStructManager(base_dir='data', is_continue=False)
        expected_out = os.path.join('data', 'monitors', 'metrics_log', 'metrics_log.json')

        metrics_group_lv1 = MetricsGroup('lv1').add(SimpleMetric(name='a', coeff=1)).add(SimpleMetric(name='b', coeff=2))
        metrics_group_lv2 = MetricsGroup('lv2').add(SimpleMetric(name='c', coeff=3))
        metrics_group_lv1.add(metrics_group_lv2)
        m = SimpleMetric(name='d', coeff=4)

        values = []
        with LogMonitor(fsm) as monitor:
            if with_final_file:
                monitor.write_final_metrics(final_file)
            for epoch in range(10):
                cur_vals = []
                for i in range(10):
                    output, target = torch.rand(1, 3), torch.rand(1, 3)
                    metrics_group_lv1.calc(output, target)
                    m._calc(output, target)
                    cur_vals.append(np.linalg.norm(output.numpy() - target.numpy()))

                values.append(float(np.mean(cur_vals)))
                monitor.set_epoch_num(epoch)
                monitor.update_metrics({'metrics': [m], 'groups': [metrics_group_lv1]})
                m.reset()
                metrics_group_lv1.reset()

        self.assertTrue(os.path.exists(expected_out) and os.path.isfile(expected_out))

        with open(expected_out, 'r') as file:
            data = json.load(file)

        self.assertIn('d', data)
        self.assertIn('lv1', data)
        self.assertIn('lv2', data['lv1'])
        self.assertIn('a', data['lv1'])
        self.assertIn('b', data['lv1'])
        self.assertIn('c', data['lv1']['lv2'])

        self.assertEqual(len(data['d']), len(values))
        self.assertEqual(len(data['lv1']['a']), len(values))
        self.assertEqual(len(data['lv1']['b']), len(values))
        self.assertEqual(len(data['lv1']['lv2']['c']), len(values))

        for i, v in enumerate(values):
            self.assertAlmostEqual(data['d'][i], values[i] * 4, delta=1e-5)
            self.assertAlmostEqual(data['lv1']['a'][i], values[i], delta=1e-5)
            self.assertAlmostEqual(data['lv1']['b'][i], values[i] * 2, delta=1e-5)
            self.assertAlmostEqual(data['lv1']['lv2']['c'][i], values[i] * 3, delta=1e-5)

        return values

    def test_metrics_processing(self):
        def check_data():
            self.assertIn('d', data)
            self.assertIn('lv1', data)
            self.assertIn('lv2', data['lv1'])
            self.assertIn('a', data['lv1'])
            self.assertIn('b', data['lv1'])
            self.assertIn('c', data['lv1']['lv2'])

            self.assertAlmostEqual(data['d'], values[-1] * 4, delta=1e-5)
            self.assertAlmostEqual(data['lv1']['a'], values[-1], delta=1e-5)
            self.assertAlmostEqual(data['lv1']['b'], values[-1] * 2, delta=1e-5)
            self.assertAlmostEqual(data['lv1']['lv2']['c'], values[-1] * 3, delta=1e-5)

        self.metrics_processing(with_final_file=False)

        shutil.rmtree('data')

        values = self.metrics_processing(with_final_file=True)
        expected_out = os.path.join('data', 'monitors', 'metrics_log', 'metrics.json')
        self.assertTrue(os.path.exists(expected_out) and os.path.isfile(expected_out))

        with open(expected_out, 'r') as file:
            data = json.load(file)

        check_data()

        shutil.rmtree('data')

        values = self.metrics_processing(with_final_file=True, final_file='my_metrics.json')
        expected_out = os.path.join('my_metrics.json')
        self.assertTrue(os.path.exists(expected_out) and os.path.isfile(expected_out))

        with open(expected_out, 'r') as file:
            data = json.load(file)

        check_data()

    def tearDown(self):
        super().tearDown()

        if os.path.exists('my_metrics.json') and os.path.isfile('my_metrics.json'):
            os.remove('my_metrics.json')
