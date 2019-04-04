"""
The main module for run inference
"""

from tqdm import tqdm
import torch

from neural_pipeline.utils import CheckpointsManager
from neural_pipeline.data_producer.data_producer import DataProducer
from neural_pipeline.data_processor import Model
from neural_pipeline.utils.fsm import FileStructManager
from neural_pipeline.data_processor.data_processor import DataProcessor


class Predictor:
    """
    Predictor run inference by training parameters

    :param model: model object, used for predict
    :param fsm: :class:`FileStructManager` object
    :param device: device for run inference
    """

    def __init__(self, model: Model, fsm: FileStructManager, device: torch.device = None):
        self._fsm = fsm
        self.__data_processor = DataProcessor(model, device=device)
        checkpoint_manager = CheckpointsManager(self._fsm)
        self.__data_processor.set_checkpoints_manager(checkpoint_manager)
        checkpoint_manager.unpack()
        self.__data_processor.load()
        checkpoint_manager.pack()

    def predict(self, data: torch.Tensor or dict):
        """
        Predict ine data

        :param data: data as :class:`torch.Tensor` or dict with key ``data``
        :return: processed output
        :rtype: model output type
        """
        return self.__data_processor.predict(data)

    def predict_dataset(self, data_producer: DataProducer, callback: callable) -> None:
        """
        Run prediction iterates by ``data_producer``

        :param data_producer: :class:`DataProducer` object
        :param callback: callback, that call for every data prediction and get it's result as parameter
        """
        loader = data_producer.get_loader()

        for img in tqdm(loader):
            callback(self.__data_processor.predict(img))
            del img
