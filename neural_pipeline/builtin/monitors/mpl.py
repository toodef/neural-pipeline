import matplotlib.pyplot as plt
import numpy as np

from neural_pipeline import AbstractMonitor


class MPLMonitor(AbstractMonitor):
    def __init__(self):
        super().__init__()

        self._realtime = True

    def update_losses(self, losses: {}):
        def on_loss(name: str, values: np.ndarray):
            plt.scatter(self.epoch_num, float(np.mean(values)), label=name)

        self._iterate_by_losses(losses, on_loss)

        if self._realtime:
            plt.pause(0.05)

    def realtime(self, is_realtime: bool) -> 'MPLMonitor':
        self._realtime = is_realtime

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.show()
