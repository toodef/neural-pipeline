import json
import os
from multiprocessing import freeze_support

import torch

from tonet.tonet.utils.file_structure_manager import FileStructManager
from .data_conveyor.data_conveyor import Dataset
from .data_processor.data_processor import DataProcessor
from .data_processor.state_manager import StateManager


class Trainer:
    def __init__(self, config_path: str, logdir_path: str = None):
        with open(config_path, 'r') as file:
            self.__config = json.load(file)

        self.__file_sruct_manager = FileStructManager(config_path, logdir_path)

        config_dir = os.path.dirname(config_path)

        with open(os.path.normpath(os.path.join(config_dir, self.__config['data_conveyor']['train']['dataset_path'])).replace("\\", "/"), 'r') as file:
            self.__train_pathes = json.load(file)
        with open(os.path.normpath(os.path.join(config_dir, self.__config['data_conveyor']['validation']['dataset_path'])).replace("\\", "/"), 'r') as file:
            self.__validation_pathes = json.load(file)

    def train(self):
        train_dataset = Dataset(self.__config['data_conveyor']['train'], self.__train_pathes, self.__file_sruct_manager)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=int(self.__config['data_conveyor']['batch_size']), shuffle=True,
                                                   num_workers=int(self.__config['data_conveyor']['threads_num']), pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            Dataset(self.__config['data_conveyor']['validation'], self.__validation_pathes, self.__file_sruct_manager),
            batch_size=int(self.__config['data_conveyor']['batch_size']), shuffle=True,
            num_workers=int(self.__config['data_conveyor']['threads_num']), pin_memory=True)

        data_processor = DataProcessor(self.__config['data_processor'], self.__file_sruct_manager, len(train_dataset.get_classes()))
        state_manager = StateManager(self.__file_sruct_manager)

        for epoch_idx in range(int(self.__config['data_conveyor']['epoch_num'])):
            data_processor.train_epoch(train_loader, val_loader, epoch_idx)
            data_processor.save_state()
            data_processor.save_weights()
            state_manager.pack()
