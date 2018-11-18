import json
import os

import torch

from neural_pipeline.utils.file_structure_manager import FileStructManager
from neural_pipeline.data_producer.builtins.segmentation import Dataset, ContoursMaskInterpreter
from neural_pipeline.data_processor.data_processor import DataProcessor
from neural_pipeline.train_pipeline.train_pipeline import AbstractTrainPipeline
from neural_pipeline.data_processor.state_manager import StateManager
import numpy as np


class Trainer:
    def __init__(self, train_pipeline: AbstractTrainPipeline, checkpoint_dir: str, network_name: str = None, logdir_path: str = None):
        self.__network_name = network_name
        self.__file_struct_manager = FileStructManager(checkpoint_dir, logdir_path)

    def train(self):
        train_dataset = Dataset(self.__config['data_producer']['train'], self.__train_pathes, ContoursMaskInterpreter(), self.__file_sruct_manager)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=int(self.__config['data_producer']['batch_size']), shuffle=True,
                                                   num_workers=int(self.__config['data_producer']['threads_num']), pin_memory=True)
        val_loader = torch.utils.data.DataLoader(Dataset(self.__config['data_producer']['validation'], self.__validation_pathes, ContoursMaskInterpreter(), self.__file_sruct_manager),
                                                 batch_size=int(self.__config['data_producer']['batch_size']), shuffle=True,
                                                 num_workers=int(self.__config['data_producer']['threads_num']), pin_memory=True)

        data_processor = DataProcessor(self.__config['data_processor'], self.__file_sruct_manager, len(train_dataset.get_classes()), network_name=self.__network_name)
        state_manager = StateManager(self.__file_sruct_manager)
        best_state_manager = StateManager(self.__file_sruct_manager, prefix="best")

        best_metric = None

        last_epoch = data_processor.get_last_epoch_idx() + 1 if data_processor.get_last_epoch_idx() > 0 else 0

        for epoch_idx in range(int(self.__config['data_producer']['epoch_num'])):
            events = data_processor.train_epoch(train_loader, val_loader, epoch_idx + last_epoch)
            data_processor.save_state()
            data_processor.save_weights()

            metric = np.mean(data_processor.get_metrics()['validation']['val_loss'])

            if best_metric is None:
                best_metric = metric
                state_manager.pack()
            elif best_metric > metric:
                print("-------------- Detect best metric --------------")
                best_metric = metric
                best_state_manager.pack()
            else:
                if events["lr_just_decreased"]:
                    state_manager.clear_files()
                    best_state_manager.unpack()
                    data_processor.load_state(best_state_manager.get_files()['state_file'])
                    data_processor.load_weights(best_state_manager.get_files()['weights_file'])
                else:
                    state_manager.pack()

            data_processor.clear_metrics()


class ProjectTrainer(Trainer):
    def __init__(self, project: Project, config_id: int, logdir_path: str = None):
        super().__init__(project.get_config_by_id(config_id), network_name=project.get_config_name_by_id(config_id), logdir_path=logdir_path)
