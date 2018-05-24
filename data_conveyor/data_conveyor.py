from random import randint

import cv2
import os


from data_conveyor.augmentations import augmentations_dict, ToPyTorch


class Dataset:
    def __init__(self, step_type: str, config: {}):
        def get_pathes(directory):
            res = []
            for cur_class in classes:
                res += [{'path': os.path.join(os.path.join(directory, str(cur_class)), file), 'target': int(cur_class) - 1}
                        for file in
                        os.listdir(os.path.join(directory, str(cur_class)))]
            return res

        dir = os.path.join(config['workdir_path'], config['data_conveyor'][step_type]['dataset_path'])
        classes = [int(d) for d in os.listdir(dir)]
        self.__pathes = get_pathes(dir)
        percentage = config['data_conveyor'][step_type]['images_percentage'] if 'images_percentage' in config['data_conveyor'][step_type] else 100
        self.__cell_size = 100 / percentage
        self.__data_num = len(self.__pathes) * percentage // 100
        self.__percentage = percentage

        self.load_augmentations(config, step_type)

        self.__augmentations_percentage = config['data_conveyor'][step_type]['augmentations_percentage'] if 'augmentations_percentage' in config['data_conveyor'][step_type] else None

    def load_augmentations(self, config: {}, step_type: str):
        """

        :param config:
        :param step_type:
        :return:
        """
        before_augmentations_config = config['data_conveyor'][step_type]['before_augmentations']
        self.__before_augmentations = [augmentations_dict[aug](before_augmentations_config) for aug in before_augmentations_config.keys()]
        if 'augmentations' in config['data_conveyor'][step_type]:
            augmentations_config = config['data_conveyor'][step_type]['augmentations']
            self.__augmentations = [augmentations_dict[aug](augmentations_config) for aug in augmentations_config.keys()]
        else:
            self.__augmentations = []
        after_augmentations_config = config['data_conveyor'][step_type]['after_augmentations']
        self.__after_augmentations = [augmentations_dict[aug](after_augmentations_config) for aug in after_augmentations_config.keys()]

    def __getitem__(self, item):
        def augmentate(image):
            for aug in self.__before_augmentations:
                image = aug(image)
            if self.__augmentations_percentage is not None and randint(1, 100) <= self.__augmentations_percentage:
                for aug in self.__augmentations:
                    image = aug(image)
            for aug in self.__after_augmentations:
                image = aug(image)
            return image

        item = randint(1, self.__cell_size) + int(item * self.__cell_size) - 1 if self.__percentage < 100 else item

        if 'target' in self.__pathes[item]:
            return {'data': augmentate(cv2.imread(self.__pathes[item]['path'])),
                    'target': self.__pathes[item]['target']}
        else:
            return augmentate(cv2.imread(self.__pathes[item]['path']))

    def __len__(self):
        return self.__data_num
