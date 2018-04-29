from random import randint

import cv2
import os

from data_conveyor.augmentations import augmentations_dict, ToPyTorch


class Dataset:
    def __init__(self, folder: str, config: {}, percentage: int = 100):
        def get_pathes(directory):
            res = []
            for cur_class in classes:
                res += [{'path': os.path.join(os.path.join(directory, str(cur_class)), file), 'target': int(cur_class)}
                        for file in
                        os.listdir(os.path.join(directory, str(cur_class)))]
            return res

        dir = os.path.join(config['workdir_path'], config['data_conveyor']['dataset_path']['folders'][folder])
        classes = [int(d) for d in os.listdir(dir)]
        self.__classes_num = len(classes)
        self.__pathes = get_pathes(dir)
        self.__cell_size = 100 / percentage
        self.__data_num = len(self.__pathes) * percentage // 100

        self.__augmentations = [augmentations_dict[aug](config) for aug in config['data_conveyor']['augmentations'].keys()]
        self.__before_output = ToPyTorch()

    def __getitem__(self, item):
        def augmentate(image):
            for aug in self.__augmentations:
                image = aug(image)
            return self.__before_output(image)

        item = randint(1, self.__cell_size) + int(item * self.__cell_size) - 1
        return {'data': augmentate(cv2.imread(self.__pathes[item]['path'])),
                'target': self.__pathes[item]['target']}

    def __len__(self):
        return self.__data_num
