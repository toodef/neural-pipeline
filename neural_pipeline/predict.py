import json
import os

from neural_pipeline.data_processor.state_manager import StateManager
from neural_pipeline.data_producer.data_producer import DataProducer
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from neural_pipeline.data_processor import Model
from neural_pipeline.utils.file_structure_manager import FileStructManager
# from neural_pipeline.data_producer.builtins.segmentation import Dataset, TiledDataset, CirclesMaskInterpreter
from neural_pipeline.data_processor.data_processor import DataProcessor


class Predictor:
    def __init__(self, model: Model, file_struct_manager: FileStructManager):
        self.__file_struct_manager = file_struct_manager
        self.__data_processor = DataProcessor(model, self.__file_struct_manager, is_cuda=True)
        state_manager = StateManager(self.__file_struct_manager)
        state_manager.unpack()
        self.__data_processor.load()
        state_manager.pack()

    def predict(self, data):
        return self.__data_processor.predict(data)

    def predict_dataset(self, data_producer: DataProducer, callback: callable):
        loader = data_producer.get_loader()

        for img in tqdm(loader):
            callback(self.__data_processor.predict(img))
            del img
