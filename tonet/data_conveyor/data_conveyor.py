import os
import sys
from abc import abstractmethod
from random import randint

import cv2
import torch

from tonet.tonet.utils.file_structure_manager import FileStructManager
from .augmentations import augmentations_dict, ToPyTorch

import numpy as np


class AbstractDataset:
    def __init__(self, config: {}, pathes: [], file_struct_manager: FileStructManager):
        self._config_path = file_struct_manager.conjfig_dir()
        self._pathes = pathes
        percentage = config['images_percentage'] if 'images_percentage' in config else 100
        self.__cell_size = 100 // percentage
        self.__data_num = len(self._pathes['data']) * percentage // 100
        self.__percentage = percentage

        self.load_augmentations(config)

        self.__augmentations_percentage = config['augmentations_percentage'] if 'augmentations_percentage' in config else None

    def get_classes(self) -> []:
        return self._pathes['labels']

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
            return cur_image, torch.from_numpy(np.expand_dims(cur_mask.astype(np.float32) / 255., 0))

        item = randint(1, self.__cell_size) + int(item * self.__cell_size) - 1 if self.__percentage < 100 else item

        data = None
        try:
            data = self._load_data(item)
        except:
            print("Image {} failed to load".format(self._get_path_by_idx(item)), file=sys.stderr)
            self._remove_item_by_idx(item)
            data_is_failed = True
            while data_is_failed:
                item = randint(1, self.__cell_size) + int(item * self.__cell_size) - 1
                try:
                    data = self._load_data(item)
                    data_is_failed = False
                except:
                    print("Image {} failed to load".format(self._get_path_by_idx(item)), file=sys.stderr)
                    self._remove_item_by_idx(item)

                if len(self) == 0:
                    print("Data is over!", file=sys.stderr)
                    return None

        if 'mask' in data:
            data, target = augmentate(data['data'], data['mask'])
            return {'data': data, 'target': target}
        else:
            return augmentate(data['data'])

    def __len__(self):
        return self.__data_num

    @abstractmethod
    def _load_data(self, index):
        pass

    @abstractmethod
    def _remove_item_by_idx(self, idx):
        pass

    @abstractmethod
    def _get_path_by_idx(self, idx):
        pass


class Dataset(AbstractDataset):
    def __init__(self, config: {}, pathes: [], file_struct_manager: FileStructManager):
        super().__init__(config, pathes, file_struct_manager)

    def _load_data(self, index: int):
        data_path = os.path.join(self._config_path, "..", "..", self._pathes['data'][index]['path']).replace("\\", "/")
        image = cv2.imread(data_path)

        if 'target' in self._pathes['data'][index]:
            cntrs = self._pathes['data'][index]['target']
            new_cntrs = np.array([np.array([[p] for p in c]) for c in cntrs])
            mask = cv2.drawContours(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), new_cntrs, -1, 255, -1)
            return {'data': image, 'mask': mask}
        else:
            return {'data': image}

    def _remove_item_by_idx(self, idx):
        del self._pathes['data'][idx]

    def _get_path_by_idx(self, idx):
        return self._pathes['data'][idx]['path']


class TiledDataset(AbstractDataset):
    class MeshGenerator:
        class MGException(Exception):
            def __init__(self, message: str):
                self.__message = message

            def __str__(self):
                return self.__message

        def __init__(self, region: list, cell_size: list, overlap: list = None):
            region = np.array(region)
            if (region[1] - region[0]).norm() == 0:
                raise self.MGException("Region size is zero!")

            if region[0][0] >= region[1][0] or region[0][1] >= region[1][1]:
                raise self.MGException("Bad region coordinates")

            if len(cell_size) < 2 or cell_size[0] <= 0 or cell_size[1] <= 0:
                raise self.MGException("Bad cell size")

            self.__region = region
            self.__cell_size = cell_size
            self.__overlap = 0.5 * np.array([overlap[0], overlap[1]] if overlap is not None else [0, 0])

        def generate_cells(self):
            result = []

            def walk_cells(callback: callable):
                y_start = self.__region[1][1] - self.__cell_size[1]
                x_start = self.__region[0][0]

                y = y_start

                step_cnt = np.array(np.abs(self.__region[1] - self.__region[0]) / self.__cell_size, dtype=np.uint64)

                for i in range(step_cnt[1]):
                    x = x_start

                    for j in range(step_cnt[0]):
                        callback([np.array([x, y]),
                                  np.array([x + self.__cell_size[0], y + self.__cell_size[1]])])
                        x += self.__cell_size[0]

                    y -= self.__cell_size[1]

            def on_cell(coords):
                coords[0] = coords[0] - self.__overlap
                coords[1] = coords[1] + self.__overlap
                result.append(coords)

            walk_cells(on_cell)
            return result

    def __init__(self, config: {}, pathes: [], file_struct_manager: FileStructManager, tile_size: list, img_original_size: list = None):
        super().__init__(config, pathes, file_struct_manager)

        if img_original_size is None:
            self.__tiles = [[tile, img_path['path']] for img_path in self._pathes['data'] for tile in self.__get_image_tiles_by_photo(img_path, tile_size)]
        else:
            self.__tiles = [[tile, img_path['path']] for img_path in self._pathes['data'] for tile in self.__get_image_tiles_by_size(img_original_size, tile_size)]

    def __get_image_tiles_by_photo(self, img_path, tile_size):
        return []

    def __get_image_tiles_by_size(self, img_original_size, tile_size: list):
        return self.MeshGenerator(img_original_size, tile_size).generate_cells()

    def _load_data(self, index):
        data_path = os.path.join(self._config_path, "..", "..", self.__tiles[index][1]).replace("\\", "/")
        [x1, x2], [y1, y2] = self.__tiles[index][0]
        image = cv2.imread(data_path)

        if 'target' in self._pathes['data'][index]:
            cntrs = self._pathes['data'][index]['target']
            new_cntrs = np.array([np.array([[p] for p in c]) for c in cntrs])
            mask = cv2.drawContours(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), new_cntrs, -1, 255, -1)
            return {'data': image[y1: y2, x1: x2, :], 'mask': mask[y1: y2, x1: x2]}
        else:
            return {'data': image[y1: y2, x1: x2, :]}

    def _remove_item_by_idx(self, idx):
        raise NotImplementedError()

    def _get_path_by_idx(self, idx):
        raise NotImplementedError()
