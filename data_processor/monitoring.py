import os

from tensorboardX import SummaryWriter

from data_processor import DataProcessor


class Monitor:
    def __init__(self):
        dir = os.path.join("workdir", "logs")
        os.makedirs(dir, exist_ok=True)
        self.__writer = SummaryWriter(dir)

    def update(self, epoch: int, metrics: {}):
        self.__update_tensorboard(epoch, metrics)

    def __update_tensorboard(self, epoch_idx: int, metrics: {}):
        for k, v in metrics.items():
            self.__writer.add_scalar('train/{}'.format(k), v, global_step=epoch_idx)

    def close(self):
        self.__writer.close()
