import os

from neural_pipeline.train_config import MetricsGroup, AbstractMetric
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

    def update_metrics(self, epoch_idx: int, metrics: {}) -> None:
        """
        Update monitor
        :param epoch_idx: current epoch index
        :param metrics: metrics dict with keys 'metrics' and 'groups'
        """
        self.__epoch_idx = epoch_idx
        self._update_metrics(epoch_idx, metrics['metrics'], metrics['groups'])

    def update_losses(self, epoch_idx: int, losses: {}) -> None:
        """
        Update monitor
        :param epoch_idx: current epoch index
        :param losses: losses values with keys 'train' and 'validation'
        """
        self.__epoch_idx = epoch_idx
        self._update_losses(epoch_idx, losses['train'], losses['validation'])

    def _update_losses(self, epoch_idx: int, train_loss: np.ndarray, val_loss: np.ndarray) -> None:
        """
        Update console
        :param epoch_idx: index of current epoch
        """
        string = "Epoch: [{}];".format(epoch_idx + 1)
        string += " {}: [{:4f}, {:4f}, {:4f}];".format('train', np.min(train_loss), np.mean(train_loss), np.max(train_loss))
        string += " {}: [{:4f}, {:4f}, {:4f}]".format('validation', np.min(val_loss), np.mean(val_loss), np.max(val_loss))
        print(string)

        if self.__writer is None:
            return

        self.__writer.add_scalars('loss', {'train': np.mean(train_loss), 'validation': np.mean(val_loss)},
                                  global_step=epoch_idx + 1)

        self.__writer.add_histogram('train/loss', np.clip(train_loss, -1, 1).astype(np.float32), global_step=epoch_idx + 1,
                                    bins=np.linspace(-1, 1, num=11).astype(np.float32))
        self.__writer.add_histogram('validation/loss', np.clip(val_loss, -1, 1).astype(np.float32), global_step=epoch_idx + 1,
                                    bins=np.linspace(-1, 1, num=11).astype(np.float32))

    def _update_metrics(self, epoch_idx: int, metrics: [AbstractMetric], metrics_groups: [MetricsGroup]) -> None:
        """
        Update console
        :param epoch_idx: index of current epoch
        :param metrics: metrics
        """

        def process_metric(cur_metric, parent_tag: str = None):
            tag = lambda name: name if parent_tag is None else '{}/{}'.format(parent_tag, name)

            if isinstance(cur_metric, MetricsGroup):
                names_dict = {m.name(): np.mean(m.get_values()) for m in cur_metric.metrics()}
                if len(names_dict) > 0:
                    self.__writer.add_scalars(tag(cur_metric.name()), names_dict, global_step=epoch_idx + 1)
                for m in cur_metric.metrics():
                    self.__writer.add_histogram(tag(m.name()),
                                                np.clip(m.get_values(), m.min_val(), m.max_val()).astype(np.float32),
                                                global_step=epoch_idx + 1,
                                                bins=np.linspace(m.min_val(), m.max_val(), num=11).astype(np.float32))
            else:
                if cur_metric.get_values().size > 0:
                    self.__writer.add_scalar(tag(cur_metric.name()), float(np.mean(cur_metric.get_values())),
                                             global_step=epoch_idx + 1)
                    self.__writer.add_histogram(tag(cur_metric.name()),
                                                np.clip(cur_metric.get_values(), cur_metric.min_val(), cur_metric.max_val()).astype(np.float32),
                                                global_step=epoch_idx + 1,
                                                bins=np.linspace(cur_metric.min_val(), cur_metric.max_val(), num=11).astype(
                                                    np.float32))

        if self.__writer is None:
            return

        for metric in metrics:
            process_metric(metric)

        for metrics_group in metrics_groups:
            for metric in metrics_group.metrics():
                process_metric(metric, metrics_group.name())
            for group in metrics_group.groups():
                process_metric(group, metrics_group.name())

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
