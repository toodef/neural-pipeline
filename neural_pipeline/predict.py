import json
import os

import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from neural_pipeline.tonet.utils.file_structure_manager import FileStructManager
from neural_pipeline.data_producer.builtins.segmentation import Dataset, TiledDataset, CirclesMaskInterpreter
from neural_pipeline.data_processor.data_processor import DataProcessor


class Predictor:
    def __init__(self, config_path: str, data_pathes: list):
        with open(config_path, 'r') as file:
            self.__config = json.load(file)

        self.__data_pathes = {'data': [{'path': p} for p in data_pathes]}

        self.__file_sruct_manager = FileStructManager(config_path)
        self.__config_dir = os.path.dirname(config_path)

    def predict(self, callback: callable):
        dataset = Dataset(self.__config['data_producer']['test'], self.__data_pathes, CirclesMaskInterpreter(), self.__file_sruct_manager)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        with open(os.path.normpath(os.path.join(self.__config_dir, self.__config['data_producer']['train']['dataset_path'])).replace("\\", "/"), 'r') as file:
            self.__train_pathes = json.load(file)
        train_dataset = Dataset(self.__config['data_producer']['train'], self.__train_pathes, CirclesMaskInterpreter(), self.__file_sruct_manager)

        self.__config['data_processor']['start_from'] = 'continue'
        data_processor = DataProcessor(self.__config['data_processor'], self.__file_sruct_manager, len(train_dataset.get_classes()))

        for img in tqdm(loader):
            callback(data_processor.predict(img).data.cpu().numpy()[0][0])
            del img

    def predict_by_tiles(self, callback: callable, tile_size: list, img_original_size: list = None):
        dataset = TiledDataset(self.__config['data_producer']['test'], self.__data_pathes, CirclesMaskInterpreter(), self.__file_sruct_manager, tile_size, img_original_size)
        # loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        with open(os.path.normpath(os.path.join(self.__config_dir, self.__config['data_producer']['train']['dataset_path'])).replace("\\", "/"), 'r') as file:
            self.__train_pathes = json.load(file)
        train_dataset = Dataset(self.__config['data_producer']['train'], self.__train_pathes, CirclesMaskInterpreter(), self.__file_sruct_manager)

        self.__config['data_processor']['start_from'] = 'continue'
        data_processor = DataProcessor(self.__config['data_processor'], self.__file_sruct_manager, len(train_dataset.get_classes()))

        # for img in tqdm(loader):
        for idx, img_tiles in tqdm(enumerate(dataset), desc="predict by tiles", leave=True):
            output_tiles = []
            image_tiles = []
            for i, tile in enumerate(img_tiles):
                image_tiles.append(dataset._load_data(i)['data'])
                output = F.sigmoid(data_processor.predict(tile.unsqueeze(0).contiguous())['output']).data.cpu().numpy()
                output_tiles.append(np.squeeze(output))
            full_output = dataset.unite_data(output_tiles, img_idx=idx)

            callback(full_output)
