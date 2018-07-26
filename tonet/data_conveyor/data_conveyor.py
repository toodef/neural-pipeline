import os
from random import randint

import cv2
import torch

from tonet.tonet.utils.file_structure_manager import FileStructManager
from .augmentations import augmentations_dict, ToPyTorch

import numpy as np


class Dataset:
    def __init__(self, config: {}, pathes: [], file_struct_manager: FileStructManager):
        self.__config_path = file_struct_manager.conjfig_dir()
        self.__pathes = pathes
        percentage = config['images_percentage'] if 'images_percentage' in config else 100
        self.__cell_size = 100 // percentage
        self.__data_num = len(self.__pathes['data']) * percentage // 100
        self.__percentage = percentage

        self.load_augmentations(config)

        self.__augmentations_percentage = config['augmentations_percentage'] if 'augmentations_percentage' in config else None

    def get_classes(self) -> []:
        return self.__pathes['labels']

    def load_augmentations(self, config: []):
        """
        Load augmentations by config
        :param config: list of augmentations config
        :return:
        """
        before_augmentations_config = config['before_augmentations']
        self.__before_augmentations = [augmentations_dict[next(iter(aug))](aug) for aug in before_augmentations_config]
        if 'augmentations' in config:
            self.__augmentations = [augmentations_dict[next(iter(aug))](aug) for aug in config['augmentations']]
        else:
            self.__augmentations = []
        after_augmentations_config = config['after_augmentations']
        self.__after_augmentations = [augmentations_dict[next(iter(aug))](aug) for aug in after_augmentations_config]

    def __getitem__(self, item):
        def augmentate(cur_image, cur_mask=None):
            def apply_augmentation_with_mask(cur_aug, img_to_augmentate, mask_to_augmentate):
                if cur_aug.is_changed_geometry():
                    return cur_aug(img_to_augmentate, mask_to_augmentate)
                else:
                    return cur_aug(img_to_augmentate), mask_to_augmentate

            def apply_augmentation(cur_aug, img_to_augmentate):
                return cur_aug(img_to_augmentate)

            for aug in self.__before_augmentations:
                if cur_mask is not None:
                    cur_image, cur_mask = apply_augmentation_with_mask(aug, cur_image, cur_mask)
                else:
                    cur_image = apply_augmentation(aug, cur_image)

            if self.__augmentations_percentage is not None and randint(1, 100) <= self.__augmentations_percentage:
                for aug in self.__augmentations:
                    try:
                        if cur_mask is not None:
                            cur_image, cur_mask = apply_augmentation_with_mask(aug, cur_image, cur_mask)
                        else:
                            cur_image = apply_augmentation(aug, cur_image)
                    except Exception as err:
                        print(aug, aug.is_changed_geometry())
                        raise err

            for aug in self.__after_augmentations:
                cur_image = apply_augmentation(aug, cur_image)

            if cur_mask is None:
                return cur_image
            return cur_image, ToPyTorch({"to_pytorch": {"percentage": 100}})(np.reshape(cur_mask.astype(np.float32) / 255, (cur_mask.shape[0], cur_mask.shape[1], 1)))

        item = randint(1, self.__cell_size) + int(item * self.__cell_size) - 1 if self.__percentage < 100 else item
        data_path = os.path.join(self.__config_path, "..", "..", self.__pathes['data'][item]['path']).replace("\\", "/")
        image = cv2.imread(data_path)
        if 'target' in self.__pathes['data'][item]:
            cntrs = self.__pathes['data'][item]['target']
            new_cntrs = np.array([np.array([[p] for p in c]) for c in cntrs])
            mask = cv2.drawContours(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), new_cntrs, -1, 255, -1)
            data, target = augmentate(image, mask)
            return {'data': data, 'target': target}
        else:
            return augmentate(image)

    def process_target(self, target):
        return np.moveaxis(np.zeros((224, 224), dtype=np.float32), -1, 0)

    def __len__(self):
        return self.__data_num
