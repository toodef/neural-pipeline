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
        :param losses: losses values dict with keys 'train' and 'validation'
        """
        pass

    def register_event(self, epoch_idx: int, text: str) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ConsoleMonitor(AbstractMonitor):
    def update_losses(self, epoch_idx: int, losses: {}):
        string = "Epoch: [{}];".format(epoch_idx + 1)
        train_loss, val_loss = losses['train'], losses['validation']
        string += " {}: [{:4f}, {:4f}, {:4f}];".format('train', np.min(train_loss), np.mean(train_loss), np.max(train_loss))
        string += " {}: [{:4f}, {:4f}, {:4f}]".format('validation', np.min(val_loss), np.mean(val_loss), np.max(val_loss))
        print(string)


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
