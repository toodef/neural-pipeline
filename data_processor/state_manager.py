import json
import os
from zipfile import ZipFile

import torch

from data_processor import DataProcessor


class StateManager:
    def __init__(self, data_processor: DataProcessor, config: {}, preffix: str = None):
        self.__data_processor = data_processor
        self.__dir = os.path.join(config['workdir_path'], config['network']['weights_dir'])
        self.__preffix = preffix

    def load(self, config: {}):
        def rm_file(file: str):
            if os.path.exists(file) and os.path.isfile(file):
                os.remove(file)

        weights_file = os.path.join(self.__dir, "weights.pth")
        state_file = os.path.join(self.__dir, "state.pth")
        result_file = os.path.join(self.__dir, self.__preffix + "_" if self.__preffix is not None else "" + "state.zip")

        with ZipFile(result_file, 'r') as zipfile:
            zipfile.extractall(self.__dir)

        config['network']['start_from'] = weights_file
        self.__data_processor = DataProcessor(config)
        self.__data_processor.load_state(state_file)
        # self.__data_processor.load_weights(state_file)
        rm_file(weights_file)
        rm_file(state_file)
        return self.__data_processor

    def save(self):
        def rm_file(file: str):
            if os.path.exists(file) and os.path.isfile(file):
                os.remove(file)

        def rename_file(file: str):
            target = file + ".old"
            rm_file(target)
            if os.path.exists(file) and os.path.isfile(file):
                os.rename(file, target)

        weights_file = os.path.join(self.__dir, "weights.pth")
        state_file = os.path.join(self.__dir, "state.pth")
        result_file = os.path.join(self.__dir, self.__preffix + "_" if self.__preffix is not None else "" + "state.zip")

        rm_file(weights_file)
        rm_file(state_file)

        self.__data_processor.save_state(state_file)
        self.__data_processor.save_weights(weights_file)

        rename_file(result_file)
        with ZipFile(result_file, 'w') as zipfile:
            zipfile.write(weights_file, os.path.basename(weights_file))
            zipfile.write(state_file, os.path.basename(state_file))

        rm_file(weights_file)
        rm_file(state_file)
