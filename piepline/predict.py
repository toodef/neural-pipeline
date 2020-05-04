"""
The main module for run inference
"""
from abc import ABCMeta

from tqdm import tqdm
import torch

from piepline.utils import CheckpointsManager
from piepline.data_producer.data_producer import DataProducer
from piepline.data_processor import Model
from piepline.utils.fsm import FileStructManager
from piepline.data_processor.data_processor import DataProcessor

__all__ = ['Predictor', 'DataProducerPredictor']


class BasePredictor(metaclass=ABCMeta):
    def __init__(self, model: Model, fsm: FileStructManager, from_best_state: bool = False):
        self._fsm = fsm
        self._data_processor = DataProcessor(model)
        checkpoint_manager = CheckpointsManager(self._fsm, prefix='best' if from_best_state else None)
        self._data_processor.set_checkpoints_manager(checkpoint_manager)
        checkpoint_manager.unpack()
        self._data_processor.load()
        checkpoint_manager.pack()


class Predictor(BasePredictor):
    """
    Predictor run inference by training parameters

    :param model: model object, used for predict
    :param fsm: :class:`FileStructManager` object
    """

    def __init__(self, model: Model, fsm: FileStructManager, from_best_state: bool = False):
        super().__init__(model, fsm, from_best_state=from_best_state)

    def predict(self, data: torch.Tensor or dict):
        """
        Predict ine data

        :param data: data as :class:`torch.Tensor` or dict with key ``data``
        :return: processed output
        :rtype: model output type
        """
        return self._data_processor.predict(data)


class DataProducerPredictor(BasePredictor):
    def __init__(self, model: Model, fsm: FileStructManager, from_best_state: bool = False):
        super().__init__(model, fsm, from_best_state=from_best_state)

    def predict(self, data_producer: DataProducer, callback: callable) -> None:
        """
        Run prediction iterates by ``data_producer``

        :param data_producer: :class:`DataProducer` object
        :param callback: callback, that call for every data prediction and get it's result as parameter
        """
        loader = data_producer.get_loader()

        for img in tqdm(loader):
            callback(self._data_processor.predict(img))
            del img
