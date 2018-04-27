import os

from tensorboardX import SummaryWriter


class Monitor:
    def __init__(self, config: {}):
        dir = os.path.join("workdir", "logs")
        dir = os.path.join(dir, "{}_{}_{:2f}_{}".format(config['network']['architecture'], config['network']['optimizer'], config['network']['learning_rate'], config['data_conveyor']['data_size'][0]))
        if os.path.exists(dir) and os.path.isdir(dir):
            idx = 0
            tmp_dir = dir + "_v{}".format(idx)
            while os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
                idx += 1
                tmp_dir = dir + "_v{}".format(idx)
            dir = tmp_dir
        os.makedirs(dir, exist_ok=True)
        self.__writer = SummaryWriter(dir)

    def update(self, epoch_idx: int, metrics: {}):
        self.__update_tensorboard(epoch_idx, metrics)
        self.__update_console(epoch_idx, metrics)

    def __update_console(self, epoch_idx: int, metrics: {}):
        string = "Epoch: {}".format(epoch_idx + 1)
        for k, v in metrics.items():
            string += "; {}: {}".format(k, v)

    def __update_tensorboard(self, epoch_idx: int, metrics: {}):
        for k, v in metrics.items():
            self.__writer.add_scalar('train/{}'.format(k), v, global_step=epoch_idx + 1)

    def close(self):
        self.__writer.close()
