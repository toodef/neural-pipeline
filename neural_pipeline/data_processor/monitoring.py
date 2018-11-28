import os

from neural_pipeline.train_config import MetricsGroup
from tensorboardX import SummaryWriter
import numpy as np

from neural_pipeline.data_processor.model import Model
from neural_pipeline.utils.file_structure_manager import FileStructManager


class Monitor:
    """
    Class, that manage metrics end events monitoring. It worked with tensorboard and console. Monitor get metrics after epoch ends and visualise it. Metrics may be float or np.array values. If
    metric is np.array - it will be shown as histogram and scalars (scalar plots contains mean valuse from array).
    """

    def __init__(self, file_struct_manager: FileStructManager, is_continue: bool, start_epoch_idx: int = 0, network_name: str = None):
        """
        :param file_struct_manager: file structure manager
        :param is_continue: is data processor continue training
        :param network_name: network name
        """
        self.__writer = None
        self.__txt_log_file = None
        self.__epoch_idx = start_epoch_idx

        dir = file_struct_manager.logdir_path()
        if dir is None:
            return

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
        self.__txt_log_file = open(os.path.join(dir, "log.txt"), 'a' if is_continue else 'w')

    def update(self, epoch_idx: int, metrics: {}) -> None:
        """
        Update monitor
        :param epoch_idx: current epoch index
        :param metrics: metrics
        """
        self.__epoch_idx = epoch_idx
        self.__update_tensorboard(epoch_idx, metrics)
        self.__update_console(epoch_idx, metrics)

    def __update_console(self, epoch_idx: int, metrics: {}) -> None:
        """
        Update console
        :param epoch_idx: index of current epoch
        :param metrics: metrics
        """
        string = "Epoch: [{}]:\n".format(epoch_idx + 1)

        for section in ['train', 'validation']:
            string += section + ':\n'
            for k, v in metrics[section].items():
                if type(v) == dict:
                    for gk, gv in v.items():
                        if type(gv) == np.ndarray and gv.size > 0:
                            string += ("; {}: [{:4f}, {:4f}, {:4f}]").format("{}_{}".format(k, gk), np.min(gv), np.mean(gv),
                                                                             np.max(gv))
                        else:
                            string += ("; {}: {:5f}" if type(gv) in [np.float64, float, np.float32] else "; {}: {}").format(
                                "{}_{}".format(k, gk), gv)
                elif type(v) == np.ndarray and v.size > 0:
                    string += ("; {}: [{:4f}, {:4f}, {:4f}]").format(k, np.min(v), np.mean(v), np.max(v))
                else:
                    string += ("; {}: {:5f}" if type(v) in [np.float64, float, np.float32] else "; {}: {}").format(k, v)
            string += '\n'
        print(string)

    def __update_tensorboard(self, epoch_idx: int, metrics: {}) -> None:
        """
        Update console
        :param epoch_idx: index of current epoch
        :param metrics: metrics
        """

        def process_metric(metric, tag: str):
            if isinstance(metric, MetricsGroup):
                self.__writer.add_scalars(tag, {m.name(): np.mean(m.get_values()) for m in metric.metrics()}, global_step=epoch_idx + 1)
                for m in metric.metrics():
                    self.__writer.add_histogram('{}_{}'.format(tag, m.name()), np.clip(v, m.min_val(), m.max_val()).astype(np.float32),
                                                global_step=epoch_idx + 1, bins=np.linspace(-0.2, 1.1, num=14).astype(np.float32))
            else:
                self.__writer.add_scalar(tag, float(np.mean(metric.get_values())), global_step=epoch_idx + 1)
                self.__writer.add_histogram('{}_{}'.format(tag, metric.name()), np.clip(v, metric.min_val(), metric.max_val()).astype(np.float32),
                                            global_step=epoch_idx + 1, bins=np.linspace(-0.2, 1.1, num=14).astype(np.float32))

        if self.__writer is None:
            return

        for metric in metrics['metrics']:
            process_metric(metric)

        for section, values in metrics.items():
            for k, v in values.items():
                if type(v) == dict:
                    for gk, gv in v.items():
                        if type(gv) == np.ndarray:
                            if gv.size > 0:
                                self.__writer.add_histogram('hist_{}/{}'.format(section, "{}_{}".format(k, gk)),
                                                            np.clip(gv, 0, 1).astype(np.float32), global_step=epoch_idx + 1,
                                                            bins=np.linspace(-0.2, 1.1, num=14).astype(np.float32))
                                self.__writer.add_scalars("plt_{}/{}".format(section, k),
                                                          {gk: process_metric(gv) for gk, gv in v.items()}, global_step=epoch_idx + 1)
                        else:
                            self.__writer.add_scalars('plt_{}/{}'.format(section, k), {gk: process_metric(gv) for gk, gv in v.items()},
                                                      global_step=epoch_idx + 1)
                elif type(v) == np.ndarray:
                    if v.size > 0:
                        self.__writer.add_histogram('hist_{}/{}'.format(section, k), np.clip(v, 0, 1).astype(np.float32),
                                                    global_step=epoch_idx + 1,
                                                    bins=np.linspace(-0.2, 1.1, num=14).astype(np.float32))
                        self.__writer.add_scalar('plt_{}/{}'.format(section, k), process_metric(v), global_step=epoch_idx + 1)
                else:
                    self.__writer.add_scalar('plt_{}/{}'.format(section, k), float(v), global_step=epoch_idx + 1)

    def write_to_txt_log(self, line: str, tag: str = None):
        self.__writer.add_text("log" if tag is None else tag, line, self.__epoch_idx)
        line = "Epoch [{}]".format(self.__epoch_idx) + ": " + line
        self.__txt_log_file.write(line + '\n')
        self.__txt_log_file.flush()

    def visualize_model(self, model: Model, tensor) -> None:
        self.__writer.add_graph(model, tensor)

    def close(self):
        if self.__txt_log_file is not None:
            self.__txt_log_file.close()
        if self.__writer is not None:
            self.__writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
