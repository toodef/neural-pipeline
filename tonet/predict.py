import cv2
import json
import os

import torch
from tqdm import tqdm

from tonet.tonet.utils.file_structure_manager import FileStructManager
from .data_conveyor.data_conveyor import Dataset
from .data_processor.data_processor import DataProcessor
from .data_processor.state_manager import StateManager


class Predictor:
    class TestDataset:
        def __init__(self, data_pathes: list):
            self.__data_pathes = data_pathes

        def __getitem__(self, item):
            return cv2.imread(self.__data_pathes[item])

        def __len__(self):
            return len(self.__data_pathes)

    def __init__(self, config_path: str, data_pathes: list):
        with open(config_path, 'r') as file:
            self.__config = json.load(file)

        self.__data_pathes = {'data': [{'path': p} for p in data_pathes]}

        self.__file_sruct_manager = FileStructManager(config_path)
        self.__config_dir = os.path.dirname(config_path)
        #
        # with open(os.path.normpath(os.path.join(config_dir, self.__config['data_conveyor']['validation']['dataset_path'])).replace("\\", "/"), 'r') as file:
        #     self.__validation_pathes = json.load(file)

    def predict(self, callback: callable):
        dataset = Dataset(self.__config['data_conveyor']['test'], self.__data_pathes, self.__file_sruct_manager)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1, shuffle=False,
                                             num_workers=1, pin_memory=True)

        with open(os.path.normpath(os.path.join(self.__config_dir, self.__config['data_conveyor']['train']['dataset_path'])).replace("\\", "/"), 'r') as file:
            self.__train_pathes = json.load(file)
        train_dataset = Dataset(self.__config['data_conveyor']['train'], self.__train_pathes, self.__file_sruct_manager)

        data_processor = DataProcessor(self.__config['data_processor'], self.__file_sruct_manager, len(train_dataset.get_classes()))
        state_manager = StateManager(self.__file_sruct_manager)

        state_manager.unpack()
        data_processor.load_state(state_manager.get_files()['state_file'])
        data_processor.load_weights(state_manager.get_files()['weights_file'])
        state_manager.clear_files()

        for img in tqdm(loader):
            callback(data_processor.predict(img))
            del img
