import json
import os

import torch

from neural_pipeline.data_processor.state_manager import StateManager
from neural_pipeline.data_producer.data_producer import DataProducer
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from neural_pipeline.data_processor import Model
from neural_pipeline.utils.file_structure_manager import FileStructManager
from neural_pipeline.data_producer.builtins.segmentation import Dataset, TiledDataset, CirclesMaskInterpreter
from neural_pipeline.data_processor.data_processor import DataProcessor


class Predictor:
    def __init__(self, model: Model, file_struct_manager: FileStructManager, data_producer: DataProducer):
        self.__data_producer = data_producer
        self.__file_struct_manager = file_struct_manager
        self.__model = model

    def predict(self, callback: callable):
        loader = self.__data_producer.get_loader()
        data_processor = DataProcessor(self.__model, self.__file_struct_manager, is_cuda=True)

        state_manager = StateManager(self.__file_struct_manager)
        state_manager.unpack()
        data_processor.load()
        state_manager.pack()

        for img in tqdm(loader):
            callback(data_processor.predict(img))
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
