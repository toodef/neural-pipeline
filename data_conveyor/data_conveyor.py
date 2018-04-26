import cv2
import os

import torchvision
import numpy as np


class Dataset:
    def __init__(self, folder: str, config: {}, transforms, percentage: int = 100):
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
        self.__pathes = self.__pathes[: len(self.__pathes) * 100 // percentage]
        self.__transforms = transforms

    def __getitem__(self, item):
        def target_to_tensor(target: int):
            tensor = np.zeros((1, self.__classes_num))
            tensor[target] = 1
            return torchvision.transforms.ToTensor()(tensor)

        return {'data': self.__transforms(torchvision.transforms.ToPILImage()(cv2.imread(self.__pathes[item]['path']))),
                'target': target_to_tensor(self.__pathes[item]['target'])}

    def __len__(self):
        return len(self.__pathes)
