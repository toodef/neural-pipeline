from abc import ABCMeta, abstractmethod

from torch.nn import Tensor


class AbstractMetricsProcessor(metaclass=ABCMeta):
    @abstractmethod
    def calculate_metrics(self, output: Tensor, target: Tensor) -> None:
        """
        Calculate metrics by output from network and target
        :param output: output from model
        :param target: target
        """


class AbstractTrainPipeline:
    def __init__(self, metrics_processor: AbstractMetricsProcessor):
        self.__metrics_processor = metrics_processor

    def metrics_processor(self) -> AbstractMetricsProcessor:
        return self.__metrics_processor
