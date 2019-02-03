import os
import numpy as np

try:
    from tensorboardX import SummaryWriter
except ImportError:
    print("Looks like tensorboardX doesn't installed. Install in via 'pip install tensorboardX' and try again")

from neural_pipeline.monitoring import AbstractMonitor
from neural_pipeline.data_processor import Model
from neural_pipeline.train_config import AbstractMetric, MetricsGroup
from neural_pipeline.utils.file_structure_manager import FileStructManager, FolderRegistrable

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class TensorboardMonitor(AbstractMonitor, FolderRegistrable):
    """
    Class, that manage metrics end events monitoring. It worked with tensorboard. Monitor get metrics after epoch ends and visualise it. Metrics may be float or np.array values. If
    metric is np.array - it will be shown as histogram and scalars (scalar plots contains mean valuse from array).

    :param file_struct_manager: file structure manager
    :param is_continue: is data processor continue training
    :param network_name: network name
    """

    def __init__(self, file_struct_manager: FileStructManager, is_continue: bool, network_name: str = None):
        super().__init__()
        self.__writer = None
        self.__txt_log_file = None

        file_struct_manager.register_dir(self)
        dir = file_struct_manager.get_path(self)
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

    def update_metrics(self, metrics: {}) -> None:
        """
        Update monitor

        :param metrics: metrics dict with keys 'metrics' and 'groups'
        """
        self._update_metrics(metrics['metrics'], metrics['groups'])

    def update_losses(self, losses: {}) -> None:
        """
        Update monitor

        :param losses: losses values with keys 'train' and 'validation'
        """
        if self.__writer is None:
            return

        def on_loss(name: str, values: np.ndarray) -> None:
            self.__writer.add_scalars('loss', {name: np.mean(values)}, global_step=self.epoch_num)
            self.__writer.add_histogram('{}/loss_hist'.format(name), np.clip(values, -1, 1).astype(np.float32),
                                        global_step=self.epoch_num, bins=np.linspace(-1, 1, num=11).astype(np.float32))

        self._iterate_by_losses(losses, on_loss)

    def _update_metrics(self, metrics: [AbstractMetric], metrics_groups: [MetricsGroup]) -> None:
        """
        Update console

        :param metrics: metrics
        """

        def process_metric(cur_metric, parent_tag: str = None):
            def add_histogram(name: str, vals, step_num, bins):
                try:
                    self.__writer.add_histogram(name, vals, step_num, bins)
                except:
                    pass

            tag = lambda name: name if parent_tag is None else '{}/{}'.format(parent_tag, name)

            if isinstance(cur_metric, MetricsGroup):
                for m in cur_metric.metrics():
                    if m.get_values().size > 0:
                        self.__writer.add_scalars(tag(m.name()), {m.name(): np.mean(m.get_values())}, global_step=self.epoch_num)
                        add_histogram(tag(m.name()) + '_hist',
                                      np.clip(m.get_values(), m.min_val(), m.max_val()).astype(np.float32),
                                      self.epoch_num, np.linspace(m.min_val(), m.max_val(), num=11).astype(np.float32))
            else:
                values = cur_metric.get_values().astype(np.float32)
                if values.size > 0:
                    self.__writer.add_scalar(tag(cur_metric.name()), float(np.mean(values)), global_step=self.epoch_num)
                    add_histogram(tag(cur_metric.name()) + '_hist',
                                  np.clip(values, cur_metric.min_val(), cur_metric.max_val()).astype(np.float32),
                                  self.epoch_num, np.linspace(cur_metric.min_val(), cur_metric.max_val(), num=11).astype(np.float32))

        if self.__writer is None:
            return

        for metric in metrics:
            process_metric(metric)

        for metrics_group in metrics_groups:
            for metric in metrics_group.metrics():
                process_metric(metric, metrics_group.name())
            for group in metrics_group.groups():
                process_metric(group, metrics_group.name())

    def update_scalar(self, name: str, value: float, epoch_idx: int = None):
        self.__writer.add_scalar(name, value, global_step=(epoch_idx if epoch_idx is not None else self.epoch_num))

    def write_to_txt_log(self, line: str, tag: str = None):
        self.__writer.add_text("log" if tag is None else tag, line, self.epoch_num)
        line = "Epoch [{}]".format(self.epoch_num) + ": " + line
        self.__txt_log_file.write(line + '\n')
        self.__txt_log_file.flush()

    def visualize_model(self, model: Model, tensor) -> None:
        self.__writer.add_graph(model, tensor)

    def close(self):
        if self.__txt_log_file is not None:
            self.__txt_log_file.close()
            self.__txt_log_file = None
            del self.__txt_log_file
        if self.__writer is not None:
            self.__writer.close()
            self.__writer = None
            del self.__writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_gir(self) -> str:
        return os.path.join('monitors', 'tensorboard')

    def get_name(self) -> str:
        return 'Tensorboard'
