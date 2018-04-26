import os

from tensorboardX import SummaryWriter


class Monitor:
    def __init__(self, config: {}):
        dir = os.path.join("workdir", "logs")
        dir = os.path.join(dir, "{}_{}_{:2f}_{}".format(config['network']['architecture'], config['network']['optimizer'], config['network']['learning_rate'], config['network']['data_size'][0]))
        os.makedirs(dir, exist_ok=True)
        self.__writer = SummaryWriter(dir)

    def update(self, epoch: int, metrics: {}):
        self.__update_tensorboard(epoch, metrics)

    def __update_tensorboard(self, epoch_idx: int, metrics: {}):
        for k, v in metrics.items():
            self.__writer.add_scalar('train/{}'.format(k), v, global_step=epoch_idx)

    def close(self):
        self.__writer.close()
