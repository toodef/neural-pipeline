import os

from tensorboardX import SummaryWriter
import numpy as np
from tonet.tonet import FileStructManager


class Monitor:
    def __init__(self, config: {}, file_struct_manager: FileStructManager, is_continue: bool, network_name: str = None):
        self.__writer = None
        dir = file_struct_manager.logdir_path()
        if dir is None:
            return

        if network_name is None:
            dir = os.path.join(dir, "{}_{}".format(config['architecture'], config['optimizer']))
        else:
            dir = os.path.join(dir, network_name)

        if not is_continue and os.path.exists(dir) and os.path.isdir(dir):
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

        for section in ['train', 'validation']:
            for k, v in metrics[section].items():
                if type(v) == dict:
                    for gk, gv in v.items():
                        if type(gv) == np.ndarray:
                            string += ("; {}: [{:4f}, {:4f}, {:4f}]").format("{}_{}".format(k, gk), np.min(gv), np.mean(gv), np.max(gv))
                        else:
                            string += ("; {}: {:5f}" if type(gv) in [np.float64, float] else "; {}: {}").format("{}_{}".format(k, gk), gv)
                elif type(v) == np.ndarray:
                    string += ("; {}: [{:4f}, {:4f}, {:4f}]").format(k, np.min(v), np.mean(v), np.max(v))
                else:
                    string += ("; {}: {:5f}" if type(v) in [np.float64, float] else "; {}: {}").format(k, v)
        print(string)

    def __update_tensorboard(self, epoch_idx: int, metrics: {}):
        def process_value(val):
            if type(val) == np.ndarray:
                return np.mean(val)
            else:
                return val

        if self.__writer is not None:
            for section, values in metrics.items():
                for k, v in values.items():
                    if type(v) == dict:
                        for gk, gv in v.items():
                            if type(gv) == np.ndarray:
                                self.__writer.add_histogram('hist_{}/{}'.format(section, "{}_{}".format(k, gk)), gv, global_step=epoch_idx + 1, bins=np.linspace(0, 1, num=10))
                        self.__writer.add_scalars("plt_{}/{}".format(section, k), {gk: process_value(gv) for gk, gv in v.items()}, global_step=epoch_idx + 1)
                    elif type(v) == np.ndarray:
                        self.__writer.add_histogram('hist_{}/{}'.format(section, k), v, global_step=epoch_idx + 1, bins=np.linspace(0, 1, num=10))
                        self.__writer.add_scalar('plt_{}/{}'.format(section, k), process_value(v), global_step=epoch_idx + 1)
                    else:
                        self.__writer.add_scalar('plt_{}/{}'.format(section, k), v, global_step=epoch_idx + 1)

    def close(self):
        self.__writer.close()
