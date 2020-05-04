from abc import ABCMeta, abstractmethod

from torch import Tensor
from torch.optim import Optimizer
from torch.nn import Module
import numpy as np
from torch.utils.data import DataLoader

try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm

from piepline.data_producer.data_producer import DataProducer
from piepline.data_processor.data_processor import TrainDataProcessor

__all__ = ['TrainConfig', 'ComparableTrainConfig', 'TrainStage', 'ValidationStage', 'AbstractMetric', 'MetricsGroup', 'MetricsProcessor', 'AbstractStage',
           'StandardStage']


class AbstractMetric(metaclass=ABCMeta):
    """
    Abstract class for metrics. When it works in piepline, it store metric value for every call of :meth:`calc`

    :param name: name of metric. Name wil be used in monitors, so be careful in use unsupported characters
    """

    def __init__(self, name: str):
        self._name = name
        self._values = np.array([])

    @abstractmethod
    def calc(self, output: Tensor, target: Tensor) -> np.ndarray or float:
        """
        Calculate metric by output from model and target

        :param output: output from model
        :param target: ground truth
        """

    def _calc(self, output: Tensor, target: Tensor):
        """
        Calculate metric by output from model and target. Method for internal use

        :param output: output from model
        :param target: ground truth
        """
        self._values = np.append(self._values, self.calc(output, target))

    def name(self) -> str:
        """
        Get name of metric

        :return: metric name
        """
        return self._name

    def get_values(self) -> np.ndarray:
        """
        Get array of metric values

        :return: array of values
        """
        return self._values

    def reset(self) -> None:
        """
        Reset array of metric values
        """
        self._values = np.array([])

    @staticmethod
    def min_val() -> float:
        """
        Get minimum value of metric. This used for correct histogram visualisation in some monitors

        :return: minimum value
        """
        return 0

    @staticmethod
    def max_val() -> float:
        """
        Get maximum value of metric. This used for correct histogram visualisation in some monitors

        :return: maximum value
        """
        return 1


class MetricsGroup:
    """
    Class for unite metrics or another :class:`MetricsGroup`'s in one namespace.
    Note: MetricsGroup may contain only 2 level of :class:`MetricsGroup`'s. So ``MetricsGroup().add(MetricsGroup().add(MetricsGroup()))``
    will raises :class:`MGException`

    :param name: group name. Name wil be used in monitors, so be careful in use unsupported characters
    """

    class MGException(Exception):
        """
        Exception for MetricsGroup
        """

        def __init__(self, msg: str):
            self.__msg = msg

        def __str__(self):
            return self.__msg

    def __init__(self, name: str):
        self.__name = name
        self.__metrics = []
        self.__metrics_groups = []
        self.__lvl = 1

    def add(self, item: AbstractMetric or 'MetricsGroup') -> 'MetricsGroup':
        """
        Add :class:`AbstractMetric` or :class:`MetricsGroup`

        :param item: object to add
        :return: self object
        :rtype: :class:`MetricsGroup`
        """
        if isinstance(item, type(self)):
            item._set_level(self.__lvl + 1)
            self.__metrics_groups.append(item)
        else:
            self.__metrics.append(item)
        return self

    def metrics(self) -> [AbstractMetric]:
        """
        Get list of metrics

        :return: list of metrics
        """
        return self.__metrics

    def groups(self) -> ['MetricsGroup']:
        """
        Get list of metrics groups

        :return: list of metrics groups
        """
        return self.__metrics_groups

    def name(self) -> str:
        """
        Get group name

        :return: name
        """
        return self.__name

    def have_groups(self) -> bool:
        """
        Is this group contains another metrics groups

        :return: True if contains, otherwise - False
        """
        return len(self.__metrics_groups) > 0

    def _set_level(self, level: int) -> None:
        """
        Internal method for set metrics group level
        TODO: if metrics group contains in two groups with different levels - this is undefined case

        :param level: parent group level
        """
        if level > 2:
            raise self.MGException("The metric group {} have {} level. There must be no more than 2 levels".format(self.__name, self.__lvl))
        self.__lvl = level
        for group in self.__metrics_groups:
            group._set_level(self.__lvl + 1)

    def calc(self, output: Tensor, target: Tensor) -> None:
        """
        Recursive calculate all metrics in this group and all nested group

        :param output: predict value
        :param target: target value
        """
        for metric in self.__metrics:
            metric._calc(output, target)
        for group in self.__metrics_groups:
            group.calc(output, target)

    def reset(self) -> None:
        """
        Recursive reset all metrics in this group and all nested group
        """
        for metric in self.__metrics:
            metric.reset()
        for group in self.__metrics_groups:
            group.reset()


class MetricsProcessor:
    """
    Collection for all :class:`AbstractMetric`'s and :class:`MetricsGroup`'s
    """

    def __init__(self):
        self._metrics = []
        self._metrics_groups = []

    def add_metric(self, metric: AbstractMetric) -> AbstractMetric:
        """
        Add :class:`AbstractMetric` object

        :param metric: metric to add
        :return: metric object
        :rtype: :class:`AbstractMetric`
        """
        self._metrics.append(metric)
        return metric

    def add_metrics_group(self, group: MetricsGroup) -> MetricsGroup:
        """
        Add :class:`MetricsGroup` object

        :param group: metrics group to add
        :return: metrics group object
        :rtype: :class:`MetricsGroup`
        """
        self._metrics_groups.append(group)
        return group

    def calc_metrics(self, output, target) -> None:
        """
        Recursive calculate all metrics

        :param output: predict value
        :param target: target value
        """
        for metric in self._metrics:
            metric.calc(output, target)
        for group in self._metrics_groups:
            group.calc(output, target)

    def reset_metrics(self) -> None:
        """
        Recursive reset all metrics values
        """
        for metric in self._metrics:
            metric.reset()
        for group in self._metrics_groups:
            group.reset()

    def get_metrics(self) -> {}:
        """
        Get metrics and groups as dict

        :return: dict of metrics and groups with keys [metrics, groups]
        """
        return {'metrics': self._metrics, 'groups': self._metrics_groups}


class AbstractStage(metaclass=ABCMeta):
    """
    Stage of training process. For example there may be 2 stages: train and validation.
    Every epochs in train loop is iteration by stages.

    :param name: name of stage
    """

    def __init__(self, name: str):
        self._name = name

    def name(self) -> str:
        """
        Get name of stage

        :return: name
        """
        return self._name

    def metrics_processor(self) -> MetricsProcessor or None:
        """
        Get metrics processor

        :return: :class:'MetricsProcessor` object or None
        """
        return None

    @abstractmethod
    def run(self, data_processor: TrainDataProcessor) -> None:
        """
        Run stage
        """

    def get_losses(self) -> np.ndarray or None:
        """
        Get losses from this stage

        :return: array of losses or None if this stage doesn't need losses
        """
        return None

    def on_epoch_end(self) -> None:
        """
        Callback for train epoch end
        """
        pass


class StandardStage(AbstractStage):
    """
    Standard stage for train process.

    When call :meth:`run` it's iterate :meth:`process_batch` of data processor by data loader

    After stop iteration ValidationStage accumulate losses from :class:`DataProcessor`.

    :param data_producer: :class:`DataProducer` object
    :param metrics_processor: :class:`MetricsProcessor`
    """

    def __init__(self, stage_name: str, is_train: bool, data_producer: DataProducer, metrics_processor: MetricsProcessor = None):
        super().__init__(name=stage_name)
        self.data_loader = None
        self.data_producer = data_producer
        self._metrics_processor = metrics_processor
        self._losses = None
        self._is_train = is_train

    def run(self, data_processor: TrainDataProcessor) -> None:
        """
        Run stage. This iterate by DataProducer and show progress in stdout

        :param data_processor: :class:`DataProcessor` object
        """
        if self.data_loader is None:
            self.data_loader = self.data_producer.get_loader()

        self._run(self.data_loader, self.name(), data_processor)

    def _run(self, data_loader: DataLoader, name: str, data_processor: TrainDataProcessor):
        with tqdm(data_loader, desc=name, leave=False) as t:
            self._losses = None
            for batch in t:
                self._process_batch(batch, data_processor)
                t.set_postfix({'loss': '[{:4f}]'.format(np.mean(self._losses))})

    def _process_batch(self, batch, data_processor: TrainDataProcessor):
        cur_loss = data_processor.process_batch(batch, metrics_processor=self.metrics_processor(), is_train=self._is_train)
        if self._losses is None:
            self._losses = cur_loss
        else:
            self._losses = np.append(self._losses, cur_loss)

    def metrics_processor(self) -> MetricsProcessor or None:
        """
        Get merics processor of this stage

        :return: :class:`MetricsProcessor` if specified otherwise None
        """
        return self._metrics_processor

    def get_losses(self) -> np.ndarray:
        """
        Get losses from this stage

        :return: array of losses
        """
        return self._losses

    def on_epoch_end(self) -> None:
        """
        Method, that calls after every epoch
        """
        self._losses = None
        metrics_processor = self.metrics_processor()
        if metrics_processor is not None:
            metrics_processor.reset_metrics()


class TrainStage(StandardStage):
    """
    Standard training stage

    When call :meth:`run` it's iterate :meth:`process_batch` of data processor by data loader with ``is_tran=True`` flag.

    After stop iteration ValidationStage accumulate losses from :class:`DataProcessor`.

    :param data_producer: :class:`DataProducer` object
    :param metrics_processor: :class:`MetricsProcessor`
    :param name: name of stage. By default 'train'
    """

    class _HardNegativesTrainStage(StandardStage):
        def __init__(self, stage_name: str, data_producer: DataProducer, part: float):
            super().__init__(stage_name, True, data_producer)
            self._part = part

        def exec(self, data_processor: TrainDataProcessor, losses: np.ndarray, indices: []) -> None:
            num_losses = int(losses.size * self._part)
            idxs = np.argpartition(losses, -num_losses)[-num_losses:]
            self._run(self.data_producer.get_loader([indices[i] for i in idxs]), self.name(), data_processor)

    def __init__(self, data_producer: DataProducer, metrics_processor: MetricsProcessor = None, name: str = 'train'):
        super().__init__(name, True, data_producer, metrics_processor)
        self.hnm = None
        self.hn_indices = []
        self._dp_pass_indices_earlier = False

    def enable_hard_negative_mining(self, part: float) -> 'TrainStage':
        """
        Enable hard negative mining. Hard negatives was taken by losses values

        :param part: part of data that repeat after train stage
        :return: self object
        """

        if not 0 < part < 1:
            raise ValueError('Value of part for hard negative mining is out of range (0, 1)')
        self.hnm = self._HardNegativesTrainStage(self.name() + '_hnm', self.data_producer, part)
        self._dp_pass_indices_earlier = self.data_producer._is_passed_indices()
        self.data_producer.pass_indices(True)
        return self

    def disable_hard_negative_mining(self) -> 'TrainStage':
        """
        Enable hard negative mining.

        :return: self object
        """
        self.hnm = None
        if not self._dp_pass_indices_earlier:
            self.data_producer.pass_indices(False)
        return self

    def run(self, data_processor: TrainDataProcessor) -> None:
        """
        Run stage

        :param data_processor: :class:`TrainDataProcessor` object
        """
        super().run(data_processor)
        if self.hnm is not None:
            self.hnm.exec(data_processor, self._losses, self.hn_indices)
            self.hn_indices = []

    def _process_batch(self, batch, data_processor: TrainDataProcessor) -> None:
        """
        Internal method for process one bathc

        :param batch: batch
        :param data_processor: :class:`TrainDataProcessor` instance
        """
        if self.hnm is not None:
            self.hn_indices.append(batch['data_idx'])
        super()._process_batch(batch, data_processor)

    def on_epoch_end(self):
        """
        Method, that calls after every epoch
        """
        super().on_epoch_end()
        if self.hnm is not None:
            self.hnm.on_epoch_end()


class ValidationStage(StandardStage):
    """
    Standard validation stage.

    When call :meth:`run` it's iterate :meth:`process_batch` of data processor by data loader with ``is_tran=False`` flag.

    After stop iteration ValidationStage accumulate losses from :class:`DataProcessor`.

    :param data_producer: :class:`DataProducer` object
    :param metrics_processor: :class:`MetricsProcessor`
    :param name: name of stage. By default 'validation'
    """

    def __init__(self, data_producer: DataProducer, metrics_processor: MetricsProcessor = None, name: str = 'validation'):
        super().__init__(name, False, data_producer, metrics_processor)


class TrainConfig:
    """
    Train process setting storage

    :param train_stages: list of stages for train loop
    :param loss: loss criterion
    :param optimizer: optimizer object
    """

    def __init__(self, model: Module, train_stages: [], loss: Module, optimizer: Optimizer):
        self._train_stages = train_stages
        self._loss = loss
        self._optimizer = optimizer
        self._model = model

    def loss(self) -> Module:
        """
        Get loss object

        :return: loss object
        """
        return self._loss

    def optimizer(self) -> Optimizer:
        """
        Get optimizer object

        :return: optimizer object
        """
        return self._optimizer

    def stages(self) -> [AbstractStage]:
        """
        Get list of stages

        :return: list of stages
        """
        return self._train_stages

    def model(self) -> Module:
        return self._model


class ComparableTrainConfig:
    """
    Train process setting storage with name. Used for train with few train configs in one time

    :param name: name of train config
    """

    def __init__(self, name: str = None):
        self._name = name

    @abstractmethod
    def get_train_config(self) -> 'TrainConfig':
        """
        Get train config

        :return: TrainConfig object
        """

    @abstractmethod
    def get_params(self) -> {}:
        """
        Get params of this config

        :return:
        """

    def get_metric_for_compare(self) -> float or None:
        """
        Get metric for compare train configs

        :return: metric value or None, if compare doesn't needed
        """
        return None
