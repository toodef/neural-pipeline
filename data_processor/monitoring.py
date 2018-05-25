import os

from tensorboardX import SummaryWriter


class Monitor:
    def __init__(self, config: {}):
        dir = os.path.join("workdir", "logs")
        dir = os.path.join(dir, "{}_{}".format(config['network']['architecture'], config['network']['optimizer']))
        if os.path.exists(dir) and os.path.isdir(dir):
            idx = 0
            tmp_dir = dir + "_v{}".format(idx)
            while os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
                idx += 1
                tmp_dir = dir + "_v{}".format(idx)
            dir = tmp_dir
        os.makedirs(dir, exist_ok=True)
        self.__writer = SummaryWriter(dir)

        self.__metrics_storage = {}
        self.__metrics_min_values = {}
        self.__metrics_max_values = {}

    def update(self, epoch_idx: int, metrics: {}):
        for k, v in metrics.items():
            if k not in self.__metrics_storage:
                self.__metrics_storage[k] = []
            if k not in self.__metrics_min_values:
                self.__metrics_min_values[k] = v
            if k not in self.__metrics_max_values:
                self.__metrics_max_values[k] = v

            self.__metrics_storage[k].append(v)
            if self.__metrics_min_values[k] >= v:
                self.__metrics_min_values[k] = v
            if self.__metrics_max_values[k] <= v:
                self.__metrics_max_values[k] = v

        self.__update_tensorboard(epoch_idx, metrics)
        self.__update_console(epoch_idx, metrics)

    def __update_console(self, epoch_idx: int, metrics: {}):
        string = "Epoch: {}".format(epoch_idx + 1)
        for k, v in metrics.items():
            string += ("; {}: {:5f}" if type(v) == float else "; {}: {}").format(k, v)
        print(string)

    def __update_tensorboard(self, epoch_idx: int, metrics: {}):
        for k, v in metrics.items():
            self.__writer.add_scalar('train/{}'.format(k), v, global_step=epoch_idx + 1)

    def close(self):
        self.__writer.close()

    def get_metric_history(self, metric_name: str, num_last_items: int = None):
        if num_last_items is not None:
            return self.__metrics_storage[metric_name][num_last_items:]
        else:
            return self.__metrics_storage[metric_name]

    def get_metrics_min_val(self, metric_name: str):
        if metric_name not in self.__metrics_min_values:
            return None
        return self.__metrics_min_values[metric_name]

    def get_metrics_max_val(self, metric_name: str):
        if metric_name not in self.__metrics_max_values:
            return None
        return self.__metrics_max_values[metric_name]
