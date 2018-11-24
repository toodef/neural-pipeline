from abc import ABCMeta, abstractmethod

from torch import Tensor
import numpy as np


class AbstractMetric(metaclass=ABCMeta):
    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def calc(self, output: Tensor, target: Tensor) -> np.ndarray:
        """
        Calculate metric by output from model and target
        :param output: output from model
        :param target: ground truth
        """

    def name(self):
        return self._name


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
            raise self.MetricsGroupException("The metric group {} have {} level. There must be no more than 2 levels"
                                             .format(self.__name, self.__lvl))
        self.__lvl = level
        for group in self.__metrics_groups:
            group.set_level(self.__lvl + 1)


class AbstractMetricsProcessor(metaclass=ABCMeta):
    def __init__(self):
        self._metrics = []
        self._metrics_groups = []
        self._values = {}

    def add_metric(self, metric: AbstractMetric):
        self._metrics.append(metric)
        self._values[metric.name()] = np.array([])

    def add_metrics_group(self, group: MetricsGroup):
        def process_metrics_group(target: dict, source_group: MetricsGroup):
            target[source_group.name()] = {}
            for metric in source_group.metrics():
                target[source_group.name()][metric.name()] = np.array([])

        self._metrics_groups.append(group)
        for metrics_group in group.groups():
            process_metrics_group(self._values[group.name()], metrics_group)
            if metrics_group.have_groups():
                for metrics_group_lv2 in metrics_group.groups():
                    if metrics_group_lv2.have_groups():
                        raise MetricsGroup.MetricsGroupException("The metric group '{}', added to '{}' have metrics groups. There "
                                                                 "must be no more than 2 levels "
                                                                 .format(metrics_group_lv2.name(), metrics_group.name()))
                    else:
                        process_metrics_group(self._values[group.name()], metrics_group_lv2)
                    self._values[group.name()][metrics_group.name()][metrics_group_lv2.name()] = np.array([])

    def calculate_metrics(self, output: Tensor, target: Tensor) -> None:
        """
        Calculate metrics by output from network and target
        :param output: output from model
        :param target: target
        """
        for metric in self._metrics:
            self._values[metric.name()] = np.append(self._values[metric.name()], metric.calc(output, target))
        for metrics_group in self._metrics_groups:
            self._values[group.name()][metrics_group.name()] = np.array([]) if metrics_group.have_groups() else {}

    @abstractmethod
    def clear_metrics(self) -> None:
        """
        Clear metrics values
        """

    def get_metrics(self):
        return self._values


class TrainPipeline:
    def __init__(self, metrics_processor: AbstractMetricsProcessor):
        self.__metrics_processor = metrics_processor

    def metrics_processor(self) -> AbstractMetricsProcessor:
        return self.__metrics_processor
