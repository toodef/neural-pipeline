from random import shuffle, randint

import cv2
import os
import numpy as np

import torchvision


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
        self.__cell_size = 100 / percentage
        self.__data_num = len(self.__pathes) * percentage // 100
        self.__transforms = transforms

    def __getitem__(self, item):
        def augmentate(img):
            def noise(img):
                row, col, ch = img.shape
                mean = 0
                var = 0.1
                sigma = var ** 0.5
                gauss = np.random.normal(mean, sigma, (row, col, ch))
                gauss = gauss.reshape(row, col, ch)
                gauss = (gauss - np.min(gauss)).astype(np.uint8)
                noisy = img.astype(np.uint8) + gauss
                return noisy

            rand_idx = randint(0, 9)
            if rand_idx > 4:
                img = cv2.flip(img, 1)
                if rand_idx > 5:
                    img = noise(img)
                if rand_idx > 6:
                    img = cv2.blur(img, (5, 5))
            return img

        item = randint(1, self.__cell_size) + int(item * self.__cell_size) - 1
        return {'data': self.__transforms(torchvision.transforms.ToPILImage()(augmentate(cv2.imread(self.__pathes[item]['path'])))),
                'target': self.__pathes[item]['target']}

    def __len__(self):
        return self.__data_num
