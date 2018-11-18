import json
import os
from abc import ABCMeta, abstractmethod


class ConfigException(Exception):
    def __init__(self, class_name):
        super(ConfigException, self).__init__()
        self.__wrong_params = []
        self.__missed_params = []
        self.__class_name = class_name

    def wrong_params(self, params: []):
        self.__wrong_params += params
        return self

    def missed_params(self, params: []):
        self.__missed_params += params
        return self

    def __str__(self):
        message = "Config contains wrong params for class: " + self.__class_name

        if str(self.__missed_params):
            message += "  Doesn't contains params: " + str(self.__missed_params)
        if str(self.__wrong_params):
            message += "  Wrong params: " + str(self.__wrong_params)


class InitedByConfig(metaclass=ABCMeta):
    def check_config(self, config: {}):
        """
        Check config is correct for this module
        :return: None
        :raise: ConfigException if config is wrong
        """
        needed_params = self._check_config()

        missed_params = []

        for p in needed_params.keys():
            if not p in config:
                missed_params.append(p)

        if len(missed_params) == 0:
            return

        raise ConfigException(type(self).__name__).missed_params(missed_params)

    @abstractmethod
    def _required_params(self):
        """
        Method, that return required parameters in config for inherit class
        :return: required parameters as dict with same structure as config dict
        """


class Config:
    def __init__(self, config_path: str = None):
        if config_path is not None:
            with open(config_path, 'r') as file:
                self.__config = json.load(file)
        else:
            self.__config = default_config

        self.__path = config_path

    def get_model_type(self):
        return self.__config['data_processor']['model_type']

    def set_model_type(self, model_type: str):
        self.__config['data_processor']['model_type'] = model_type

    def get_data_size(self):
        return self.__config['data_processor']['data_size']

    def set_data_size(self, data_size: list):
        self.__config['data_processor']['data_size'] = data_size

    def get_config(self):
        return self.__config

    def get_config_path(self):
        return self.__path


class Project:
    def __init__(self, project_path: str = None):
        if project_path is not None:
            with open(project_path, 'r') as file:
                self.__project = json.load(file)
        else:
            self.__project = default_config

        self.__project_dir = os.path.dirname(project_path)

    def get_config_list(self):
        return self.__project

    def get_config_by_id(self, idx: int):
        return Config(os.path.join(self.__project_dir, "workdir", str(self.__project[idx]['id']), "config.json"))

    def get_config_name_by_id(self, idx: int):
        return self.__project[idx]['name']


default_config = {
    "data_processor": {
        'model_type': "u_net",
        "data_size": [224, 224, 3],
        "architecture": "densenet201",
        "optimizer": "Adam",
        "learning_rate": {"start_value": 0.001,
                          "steps_before_decrease": 10,
                          "decrease_coefficient": 10},
        "start_from": "url"
    },
    "data_producer": {
        "batch_size": 1,
        "threads_num": 1,
        "epoch_num": 1,
        "train_by_folds": False,
        "train": {
            "dataset_path": "train.json",
            "before_augmentations": [{"resize": {"percentage": 100, "size": 224}},
                                     {"ccrop": {"percentage": 100, "size": 224}}],
            "augmentations": [{"hflip": {"percentage": 0}},
                              {"rrotate": {"percentage": 0, "interval": [-10, 10]}},
                              {"rbrightness": {"percentage": 0, 'interval': [10, 100]}},
                              {"rcontrast": {"percentage": 0, 'interval': [50, 150]}},
                              {"gauss_noise": {"percentage": 0, 'mean': 1, 'var': 0.01, 'interval': 50}},
                              {"snp_noise": {"percentage": 0, 's_vs_p': 0.5, 'amount': 0.1}},
                              {"blur": {"percentage": 100, 'ksize': (3, 3)}}],
            "after_augmentations": [{"to_pytorch": {"percentage": 100}}, {"normalize": {"percentage": 100}}],
            "augmentations_percentage": 100,
            "images_percentage": 1
        },
        "validation": {
            "dataset_path": "validation.json",
            "before_augmentations": [{"resize": {"percentage": 100, "size": [224, 224]}}],
            "after_augmentations": [{"to_pytorch": {"percentage": 100}, "normalize": {"percentage": 100}}],
            "augmentations_percentage": 100,
            "images_percentage": 1
        },
        "test": {
            "dataset_path": "test.json",
            "before_augmentations": [{"resize": {"percentage": 100, "size": 224}},
                                     {"ccrop": {"percentage": 100, "size": 224}}],
            "augmentations_percentage": 100,
            "images_percentage": 100
        }
    }
}
