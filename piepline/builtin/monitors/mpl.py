"""
This module contains Matplotlib monitor interface
"""

from random import shuffle

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError:
    import sys
    print("Can't import Matplotlib in module neural-pipeline.builtin.mpl. Try perform 'pip install matplotlib'", file=sys.stderr)
    sys.exit(1)

import numpy as np

from piepline import AbstractMonitor
from piepline.train_config import MetricsGroup


class MPLMonitor(AbstractMonitor):
    """
    This monitor show all data in Matplotlib plots
    """
    class _Plot:
        __cmap = plt.cm.get_cmap('hsv', 10)
        __cmap_indices = [i for i in range(10)]
        shuffle(__cmap_indices)

        def __init__(self, names: [str]):
            self._handle = names[0]

            self._prev_values = {}
            self._colors = {}
            self._axis = None

        def add_values(self, values: {}, epoch_idx: int) -> None:
            for n, v in values.items():
                self.add_value(n, v, epoch_idx)

        def add_value(self, name: str, val: float, epoch_idx: int) -> None:
            if name not in self._prev_values:
                self._prev_values[name] = None
                self._colors[name] = self.__cmap(self.__cmap_indices[len(self._colors)])
            prev_value = self._prev_values[name]
            if prev_value is not None and self._axis is not None:
                self._axis.plot([prev_value[1], epoch_idx], [prev_value[0], val], label=name, c=self._colors[name])
            self._prev_values[name] = [val, epoch_idx]

        def place_plot(self, axis) -> None:
            self._axis = axis

            for n, v in self._prev_values.items():
                self._axis.scatter(v[1], v[0], label=n, c=self._colors[n])

            self._axis.set_ylabel(self._handle)
            self._axis.set_xlabel('epoch')
            self._axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            self._axis.legend()
            plt.grid()

    def __init__(self):
        super().__init__()

        self._realtime = True
        self._plots = {}
        self._plots_placed = False

    def update_losses(self, losses: {}):
        def on_loss(name: str, values: np.ndarray):
            plot = self._cur_plot(['loss', name])
            plot.add_value(name, np.mean(values), self.epoch_num)

        self._iterate_by_losses(losses, on_loss)

        if not self._plots_placed:
            self._place_plots()
            self._plots_placed = True

        if self._realtime:
            plt.pause(0.01)

    def update_metrics(self, metrics: {}) -> None:
        for metric in metrics['metrics']:
            self._process_metric(metric)

        for metrics_group in metrics['groups']:
            for metric in metrics_group.metrics():
                self._process_metric(metric, metrics_group.name())
            for group in metrics_group.groups():
                self._process_metric(group)

    def realtime(self, is_realtime: bool) -> 'MPLMonitor':
        """
        Is need to show data updates in realtime

        :param is_realtime: is need realtime
        :return: self object
        """
        self._realtime = is_realtime

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.show()

    def _process_metric(self, cur_metric, parent_tag: str = None):
        if isinstance(cur_metric, MetricsGroup):
            for m in cur_metric.metrics():
                names = self._compile_names(parent_tag, [cur_metric.name(), m.name()])
                plot = self._cur_plot(names)
                if m.get_values().size > 0:
                    plot.add_value(m.name(), np.mean(m.get_values), self.epoch_num)
        else:
            values = cur_metric.get_values().astype(np.float32)
            names = self._compile_names(parent_tag, [cur_metric.name()])
            plot = self._cur_plot(names)
            if values.size > 0:
                plot.add_value(cur_metric.name(), np.mean(values), self.epoch_num)

    @staticmethod
    def _compile_names(parent_tag: str, names: [str]):
        if parent_tag is not None:
            return [parent_tag] + names
        else:
            return names

    def _cur_plot(self, names: [str]) -> '_Plot':
        if names[0] not in self._plots:
            self._plots[names[0]] = self._Plot(names)
        return self._plots[names[0]]

    def _place_plots(self):
        number_of_subplots = len(self._plots)
        idx = 1
        for n, v in self._plots.items():
            v.place_plot(plt.subplot(number_of_subplots, 1, idx))
            idx += 1
