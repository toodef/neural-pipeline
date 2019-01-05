from abc import ABCMeta, abstractmethod

from torch import Tensor
from torch.optim import Optimizer
from torch.nn import Module
import numpy as np
from tqdm import tqdm

from neural_pipeline.data_producer.data_producer import DataProducer
from neural_pipeline.data_processor.data_processor import TrainDataProcessor

__all__ = ['AbstractMetric', 'MetricsGroup', 'AbstractMetricsProcessor', 'AbstractLearningRate', 'AbstractStage', 'TrainStage', 'ValidationStage', 'TrainConfig']


class AbstractMetric(metaclass=ABCMeta):
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
        Calculate metric by output from model and target
        :param output: output from model
        :param target: ground truth
        """
        self._values = np.append(self._values, self.calc(output, target))

    def name(self):
        return self._name

    def get_values(self) -> np.ndarray:
        return self._values

    def reset(self) -> None:
        self._values = np.array([])

    @staticmethod
    def min_val() -> float:
        return 0

    @staticmethod
    def max_val() -> float:
        return 1


class MetricsGroup:
    class MetricsGroupException(Exception):
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
        if isinstance(item, type(self)):
            item.set_level(self.__lvl + 1)
            self.__metrics_groups.append(item)
        else:
            self.__metrics.append(item)
        return self

    def metrics(self) -> [AbstractMetric]:
        return self.__metrics

    def groups(self) -> ['MetricsGroup']:
        return self.__metrics_groups

    def name(self) -> str:
        return self.__name

    def have_groups(self) -> bool:
        return len(self.__metrics_groups) > 0

    def set_level(self, level: int):
        if level > 2:
            raise self.MetricsGroupException("The metric group {} have {} level. There must be no more than 2 levels".format(self.__name, self.__lvl))
        self.__lvl = level
        for group in self.__metrics_groups:
            group.set_level(self.__lvl + 1)

    def calc(self, output: Tensor, target: Tensor):
        for metric in self.__metrics:
            metric._calc(output, target)
        for group in self.__metrics_groups:
            group.calc(output, target)

    def reset(self):
        for metric in self.__metrics:
            metric.reset()
        for group in self.__metrics_groups:
            group.reset()


class AbstractMetricsProcessor(metaclass=ABCMeta):
    def __init__(self):
        self._metrics = []
        self._metrics_groups = []

    def add_metric(self, metric: AbstractMetric) -> AbstractMetric:
        self._metrics.append(metric)
        return metric

    def add_metrics_group(self, group: MetricsGroup) -> MetricsGroup:
        self._metrics_groups.append(group)
        return group

    @abstractmethod
    def calc_metrics(self, output, target) -> None:
        """
        Calculate metrics by output from network and target
        :param output: output from model
        :param target: target
        """
        for metric in self._metrics:
            metric.calc(output, target)
        for group in self._metrics_groups:
            group.calc(output, target)

    def reset_metrics(self) -> None:
        """
        Reset metrics values
        """
        for metric in self._metrics:
            metric.reset()
        for group in self._metrics_groups:
            group.reset()

    def get_metrics(self) -> {}:
        return {'metrics': self._metrics, 'groups': self._metrics_groups}


class AbstractLearningRate:
    """
    Learning rate manage strategy.
    This class provide lr decay by loss values. If loss doesn't update minimum throw defined number of steps - lr decay to defined coefficient
    """

    def __init__(self):
        self._value = 1e-4

    def value(self) -> float:
        """
        Get value of current learning rate
        """
        return self._value

    def set_value(self, value):
        self._value = value


class AbstractStage(metaclass=ABCMeta):
    def __init__(self, name: str):
        self._name = name

    def name(self) -> str:
        return self._name

    def metrics_processor(self) -> AbstractMetricsProcessor or None:
        return None

    @abstractmethod
    def run(self, data_processor: TrainDataProcessor) -> None:
        """
        Run stage
        """


class TrainStage(AbstractStage):
    def __init__(self, data_producer: DataProducer, metrics_processor: AbstractMetricsProcessor):
        super().__init__(name='train')
        self.data_loader = data_producer.get_loader()
        self._metrics_processor = metrics_processor

    def run(self, data_processor: TrainDataProcessor) -> None:
        with tqdm(self.data_loader, desc=self.name(), leave=False) as t:
            for batch in t:
                losses = data_processor.process_batch(batch, metrics_processor=self.metrics_processor(), is_train=True)
                t.set_postfix({'loss': '[{:4f}]'.format(np.mean(losses))})

    def metrics_processor(self) -> AbstractMetricsProcessor or None:
        return self._metrics_processor


class ValidationStage(AbstractStage):
    def __init__(self, data_producer: DataProducer, metrics_processor: AbstractMetricsProcessor):
        super().__init__(name='validation')
        self.data_loader = data_producer.get_loader()
        self._metrics_processor = metrics_processor

    def run(self, data_processor: TrainDataProcessor) -> None:
        with tqdm(self.data_loader, desc=self.name(), leave=False) as t:
            for batch in t:
                losses = data_processor.process_batch(batch, metrics_processor=self.metrics_processor(), is_train=False)
                t.set_postfix({'loss': '[{:4f}]'.format(np.mean(losses))})

    def metrics_processor(self) -> AbstractMetricsProcessor or None:
        return self._metrics_processor


class TrainConfig:
    def __init__(self, train_stages: [], loss: Module, optimizer: Optimizer, experiment_name: str = None):
        self._train_stages = train_stages
        self.__loss = loss
        self.__experiment_name = experiment_name
        self.__optimizer = optimizer

    def loss(self):
        return self.__loss

    def optimizer(self):
        return self.__optimizer

    def experiment_name(self):
        return self.__experiment_name

    def stages(self) -> [AbstractStage]:
        return self._train_stages
