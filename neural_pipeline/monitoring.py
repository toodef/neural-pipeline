from abc import ABCMeta
import numpy as np

__all__ = ['MonitorHub', 'AbstractMonitor', 'ConsoleMonitor']


class AbstractMonitor(metaclass=ABCMeta):
    def update_metrics(self, epoch_idx, metrics) -> None:
        """
        Update monitor
        :param epoch_idx: current epoch index
        :param metrics: metrics dict with keys 'metrics' and 'groups'
        """
        pass

    def update_losses(self, epoch_idx: int, losses: {}) -> None:
        """
        Update monitor

        :param epoch_idx: current epoch index
        :param losses: losses values dict with keys is names of stages in train pipeline (e.g. [train, validation])
        """
        pass

    @staticmethod
    def _iterate_by_losses(losses: {}, callback: callable) -> None:
        """
        Internal method for unify iteration by losses dict

        :param losses: dic of losses
        :param callback: callable, that call for every loss value and get params loss_name and loss_values: ``callback(name: str, values: np.ndarray)``
        """
        for m, v in losses.items():
            callback(m, v)

    def register_event(self, epoch_idx: int, text: str) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ConsoleMonitor(AbstractMonitor):
    class ResStr:
        def __init__(self, start: str):
            self.res = start

        def append(self, string: str):
            self.res += string

        def __str__(self):
            return self.res[:len(self.res) - 1]

    def update_losses(self, epoch_idx: int, losses: {}) -> None:
        def on_loss(name: str, values: np.ndarray, string) -> None:
            string.append(" {}: [{:4f}, {:4f}, {:4f}];".format(name, np.min(values), np.mean(values), np.max(values)))

        res_string = self.ResStr("Epoch: [{}];".format(epoch_idx + 1))
        self._iterate_by_losses(losses, lambda m, v: on_loss(m, v, res_string))
        print(res_string)


class MonitorHub(AbstractMonitor):
    def __init__(self):
        self.monitors = []

    def add_monitor(self, monitor: AbstractMonitor):
        self.monitors.append(monitor)

    def update_metrics(self, epoch_idx: int, metrics: {}) -> None:
        """
        Update monitor
        :param epoch_idx: current epoch index
        :param metrics: metrics dict with keys 'metrics' and 'groups'
        """
        for m in self.monitors:
            m.update_metrics(epoch_idx, metrics)

    def update_losses(self, epoch_idx: int, losses: {}) -> None:
        """
        Update monitor
        :param epoch_idx: current epoch index
        :param losses: losses values with keys 'train' and 'validation'
        """
        for m in self.monitors:
            m.update_losses(epoch_idx, losses)

    def register_event(self, epoch_idx: int, text: str) -> None:
        for m in self.monitors:
            m.register_event(epoch_idx, text)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for m in self.monitors:
            m.__exit__(exc_type, exc_val, exc_tb)
