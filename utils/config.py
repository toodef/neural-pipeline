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


default_config = {
    "data_processor": {
        "architecture": "densenet201",
        "optimizer": "Adam",
        "learning_rate": {"start_value": 0.001,
                          "skip_steps_number": 2,
                          "decrease_coefficient": 10,
                          "first_epoch_decrease_coeff": 10},
        "start_from": "url"
    },
    "data_conveyor": {
        "data_size": [224, 224, 3],
        "batch_size": 1,
        "threads_num": 1,
        "epoch_num": 1,
        "train": {
            "dataset_path": "train",
            "before_augmentations": {"resize": {"percentage": 100, "size": 224},
                                     "ccrop": {"percentage": 100, "size": 224}},
            "augmentations": {"hflip": {"percentage": 0},
                              "rrotate": {"percentage": 0, "interval": [-10, 10]},
                              "rbrightness": {"percentage": 0, 'interval': [10, 100]},
                              "rcontrast": {"percentage": 0, 'interval': [50, 150]},
                              "gauss_noise": {"percentage": 0, 'mean': 1, 'var': 0.01, 'interval': 50},
                              "snp_noise": {"percentage": 0, 's_vs_p': 0.5, 'amount': 0.1},
                              "blur": {"percentage": 100, 'ksize': (3, 3)}},
            "after_augmentations": {"to_pytorch": {"percentage": 100}, "normalize": {"percentage": 100}},
            "augmentations_percentage": 100,
            "images_percentage": 1
        },
        "validation": {
            "dataset_path": "validation",
            "before_augmentations": {"resize": {"percentage": 100, "size": [224, 224]}},
            "after_augmentations": {"to_pytorch": {"percentage": 100}, "normalize": {"percentage": 100}},
            "augmentations_percentage": 100,
            "images_percentage": 1
        },
        "test": {
            "dataset_path": "test",
            "before_augmentations": {"resize": {"percentage": 100, "size": 224},
                                    "ccrop": {"percentage": 100, "size": 224}},
            "augmentations_percentage": 100,
            "images_percentage": 100
        }

    },
    "workdir_path": "workdir"
}
