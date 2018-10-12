import os
import sys
from abc import abstractmethod, ABCMeta
from random import randint

import cv2
import torch

from neural_pipeline.tonet.utils.file_structure_manager import FileStructManager
from .augmentations import augmentations_dict

import numpy as np


class AbstractMaskInterpreter(metaclass=ABCMeta):
    """
    Abstract interface to mask interpreters
    Different kinds of masks interpreters need to correct interpret contours, circles, lines and other
    """

    @abstractmethod
    def __call__(self, shape, data):
        """
        Convert shape to bitwise mask
        :param shape: list of points
        :param data: image
        :return:
        """
        pass


class ContoursMaskInterpreter(AbstractMaskInterpreter):
    def __call__(self, shape, contours: []):
        new_cntrs = np.array([np.array([[p] for p in c]) for c in contours])
        return cv2.drawContours(np.zeros((shape[0], shape[1]), dtype=np.uint8), new_cntrs, -1, 255, -1)


class CirclesMaskInterpreter(AbstractMaskInterpreter):
    def __call__(self, shape, circles: []):
        res_mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
        for c in circles:
            res_mask = cv2.circle(res_mask, (c[0], c[1]), c[2], 255, -1)
        return res_mask


mask_interpreters = {'cntr': ContoursMaskInterpreter, 'circ': CirclesMaskInterpreter}


class AbstractDataset:
    """
    Abstract class for every dataset
    Dataset manage:
    1) Data loading
    2) Generation targets (masks) for data
    3) Data augmentation
    This not full driven data loading order. This just callback for torch.utils.data.DataLoader. See use samples in tonet.train.
    Augmentation divide to 3 groups:
    1) Before augmentations - preprocess. resize, central crop, random crop for example
    2) Augmentations - classic expectation of augmentations
    3) After augmentations - postprocess. normalize, cast to porch tensor for example
    """

    def __init__(self, config: {}, pathes: [], file_struct_manager: FileStructManager):
        """
        :param config: dataset config
        :param pathes: dataset date. Like pathes, targets, labels
        :param file_struct_manager: file_struct_manager
        """
        self._config_path = file_struct_manager.config_dir()
        self._pathes = pathes
        percentage = config['images_percentage'] if 'images_percentage' in config else 100
        self.__cell_size = 100 // percentage
        self.__percentage = percentage

        self.load_augmentations(config)

        self.__augmentations_percentage = config['augmentations_percentage'] if 'augmentations_percentage' in config else None

        self.__indices = None

    def get_classes(self) -> []:
        """
        Get classes (labels) from dataset config
        :return: list of dicts
        """
        return self._pathes['labels']

    def load_augmentations(self, config: []) -> None:
        """
        Load augmentations by config
        :param config: list of augmentations config
        """
        before_augmentations_config = config['before_augmentations']
        self.__before_augmentations = [augmentations_dict[next(iter(aug))](aug) for aug in before_augmentations_config]
        if 'augmentations' in config:
            self.__augmentations = [augmentations_dict[next(iter(aug))](aug) for aug in config['augmentations']]
        else:
            self.__augmentations = []
        after_augmentations_config = config['after_augmentations']
        self.__after_augmentations = [augmentations_dict[next(iter(aug))](aug) for aug in after_augmentations_config]

    def __getitem__(self, item) -> torch.Tensor or {}:
        """
        Load data from dataset by index
        :param item: index of required data
        :return: data or data and target in dict
        """

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

        if self.__indices is None:
            item = randint(1, self.__cell_size) + int(item * self.__cell_size) - 1 if self.__percentage < 100 else item
        else:
            item = self.__indices[item]

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

    def _generate_target(self, image, index):
        if 'target' in self._pathes['data'][index]:
            masks = np.zeros_like((image.shape[0], image.shape[1], len(self.get_classes())))
            for item in self._pathes['data'][index]['target']:
                label = item['lab']
                data = item['dat']
                mask_interpreter = mask_interpreters[item['interp']]((image.shape[0], image.shape[1]), data)
                masks[:, :, label] = mask_interpreter()
            return {'data': image, 'mask': np.squeeze(masks)}
        else:
            return {'data': image}

    def __len__(self):
        """
        Get actual data number
        :return: data number
        """
        if self.__indices is None:
            return self._get_data_number() * self.__percentage // 100
        return len(self.__indices)

    @abstractmethod
    def _get_data_number(self) -> int:
        """
        Get number of data in dataset
        """
        pass

    @abstractmethod
    def _load_data(self, index) -> np.array:
        """
        Load data by index
        :param index: index of data
        :return: data
        """
        pass

    @abstractmethod
    def _remove_item_by_idx(self, idx) -> None:
        """
        Remove data from loading list by index
        :param idx: index
        """
        pass

    @abstractmethod
    def _get_path_by_idx(self, idx) -> np.array:
        """
        Get data path by index
        :param idx: index
        :return: data
        """
        pass

    def walk_by_indices(self, indices: list) -> None:  # TODO: is this really needed
        """
        Haven't ideas what is it. Maybe needed
        :param indices: list of indices
        """
        self.__indices = indices

    def clear_indices(self):
        """
        Clear all indices
        :return:
        """
        self.__indices = None


class Dataset(AbstractDataset):
    """
    Classic dataset, that just load images from disk
    """

    def __init__(self, config: {}, pathes: [], mask_interpreter: AbstractMaskInterpreter, file_struct_manager: FileStructManager):
        """
        :param config: dataset config
        :param pathes: dataset date. Like pathes, targets, labels
        :param mask_interpreter: concrette mask interpreter
        :param file_struct_manager: file_struct_manager
        """
        super().__init__(config, pathes, mask_interpreter, file_struct_manager)

    def _load_data(self, index: int):
        data_path = self._get_path_by_idx(index)
        image = cv2.imread(data_path, -1)
        return self._generate_target(image, index)

    def _get_data_number(self):
        return len(self._pathes['data'])

    def _remove_item_by_idx(self, idx):
        del self._pathes['data'][idx]

    def _get_path_by_idx(self, idx):
        return os.path.join(self._config_path, "..", "..", self._pathes['data'][idx]['path']).replace("\\", "/")


class TiledDataset(AbstractDataset):
    """
    Dataset, that divide images by tiles and return tile by request
    """

    class MeshGenerator:
        """
        Class, that generate tiles by image
        """

        class MGException(Exception):
            def init(self, message: str):
                self.message = message

            def __str(self):
                return self.message

        def __init__(self, region: list, cell_size: list):
            """
            :param region: size of image
            :param cell_size: size of tile
            """
            region = np.array(region)
            if np.linalg.norm(region[1] - region[0]) == 0:
                raise self.MGException("Region size is zero!")

            if region[0][0] >= region[1][0] or region[0][1] >= region[1][1]:
                raise self.MGException("Bad region coordinates")

            if len(cell_size) < 2 or cell_size[0] <= 0 or cell_size[1] <= 0:
                raise self.MGException("Bad cell size")

            self.__region = np.array(region, dtype=np.uint64)
            self.__cell_size = np.array(cell_size, dtype=np.uint64)

        def generate_cells(self):
            result = []

            def walk_cells(callback: callable):
                x_start, y_start = self.__region[0][0], self.__region[0][1]

                y = y_start

                cells_cnt = np.ceil(np.abs(self.__region[1] - self.__region[0]) / self.__cell_size).astype(np.uint32)
                global_offset = (cells_cnt * self.__cell_size - self.__region[1])

                offset = (global_offset / (cells_cnt - np.array([1, 1]))).astype(np.uint32)
                mod_offset = np.mod(global_offset, (cells_cnt - np.array([1, 1]))).astype(np.uint32)

                def calc_offset(axis):
                    # res = offset[axis] - (mod_offset[axis] > 0)
                    # mod_offset[axis] = mod_offset[axis] - (1 if mod_offset[axis] > 0 else 0)
                    return offset[axis]

                x_offsets = [0] + [calc_offset(0) for i in range(int(cells_cnt[0] - 2))] + [0]
                y_offsets = [0] + [calc_offset(1) for i in range(int(cells_cnt[1] - 2))] + [0]

                for i in range(int(cells_cnt[1])):
                    x = x_start
                    for j in range(int(cells_cnt[0])):
                        cur_offset = [x_offsets[j], y_offsets[i]]
                        # cur_offset = offset * np.array([0 < j < (step_cnt[0] - 2), 0 < i < (step_cnt[1] - 2)], dtype=np.uint64)
                        callback([np.array([x, y]), np.array([x + self.__cell_size[0], y + self.__cell_size[1]])], cur_offset)
                        if j < (cells_cnt[0] - 2):
                            x += self.__cell_size[0] - (offset[0] * (0 < j))
                        else:
                            x = self.__region[1][0] - self.__cell_size[0]

                    if i < (cells_cnt[1] - 2):
                        y += self.__cell_size[1] - (offset[1] * (0 < i))
                    else:
                        y = self.__region[1][1] - self.__cell_size[1]

            def on_cell(coords, cur_offset):
                result.append((np.array(coords, dtype=np.uint32) - cur_offset).astype(np.uint32))

            walk_cells(on_cell)
            return result

    def __init__(self, config: {}, pathes: [], mask_interpreter: AbstractMaskInterpreter, file_struct_manager: FileStructManager, tile_size: list, img_original_size: list = None):
        """
        :param config: dataset config
        :param pathes: dataset date. Like pathes, targets, labels
        :param mask_interpreter: concrette mask interpreter
        :param file_struct_manager: file_struct_manager
        :param tile_size: size of tiles
        :param img_original_size: images size (optional)
        """

        super().__init__(config, pathes, mask_interpreter, file_struct_manager)

        self.__tiles = []
        self.__images = []
        if img_original_size is None:
            for i, img_path in enumerate(self._pathes['data']):
                img = cv2.imread(img_path['path'])
                img_size = [img.shape[0], img.shape[1]]
                cur_tiles = []
                for tile in self.__get_image_tiles_by_size([img_size[1], img_size[0]], tile_size):
                    cur_tiles.append({"tile": tile, "img_id": i})
                img_info = {"path": img_path['path'], "tiles_ids": list(range(len(self.__tiles), len(self.__tiles) + len(cur_tiles))), "size": img_size}
                if "target" in img_path:
                    img_info["target"] = img_path['target']
                self.__images.append(img_info)
                self.__tiles.extend(cur_tiles)
            # self.__tiles = [{"tile": tile, "path": img_path['path'], "target": img_path['target']} if "target" in img_path else {"tile": tile, "path": img_path['path']} for img_path in self._pathes['data'] for tile in self.__get_image_tiles_by_photo(img_path, tile_size)]
        else:
            one_image_tiles = self.__get_image_tiles_by_size(img_original_size, tile_size)
            for i, img_path in enumerate(self._pathes['data']):
                cur_tiles = []
                for tile in one_image_tiles:
                    cur_tiles.append({"tile": tile, "img_id": i})
                img_info = {"path": img_path['path'], "tiles_ids": list(range(len(self.__tiles), len(self.__tiles) + len(cur_tiles))), "size": img_original_size}
                if "target" in img_path:
                    img_info["target"] = img_path['target']
                self.__images.append(img_info)
                # self.__images.append({"path": img_path['path'], "tiles": cur_tiles, "target": img_path['target']} if "target" in img_path else {"path": img_path['path'], "tiles": cur_tiles})
                self.__tiles.extend(cur_tiles)
            # self.__tiles = [{"tile": tile, "path": img_path['path'], "target": img_path['target']} if "target" in img_path else {"tile": tile, "path": img_path['path']} for img_path in self._pathes['data'] for tile in self.__get_image_tiles_by_size(img_original_size, tile_size)]

    def __get_image_tiles_by_size(self, img_original_size, tile_size: list) -> []:
        """
        Calculate tiles by image size
        :param img_original_size: image size
        :param tile_size: tile size
        :return: list of tiles
        """
        return self.MeshGenerator(np.array([[0, 0], img_original_size]), tile_size).generate_cells()

    def _load_data(self, index) -> {}:
        """
        Load data by index
        :param index:
        :return:
        """
        data_path = self._get_path_by_idx(self.__tiles[index]['img_id'])
        [x1, y1], [x2, y2] = self.__tiles[index]["tile"]
        image = cv2.imread(data_path, -1)

        res = self._generate_target(image, index)
        return {'data': res['data'][y1: y2, x1: x2, :], 'mask': res['mask'][y1: y2, x1: x2]} if 'mask' in res else {'data': res['data'][y1: y2, x1: x2, :]}

    def _get_data_number(self):
        return len(self.__tiles)

    def _remove_item_by_idx(self, idx):
        del self.__tiles[idx]

    def _get_path_by_idx(self, idx):
        return os.path.join(self._config_path, "..", "..", self.__images[idx]["path"]).replace("\\", "/")

    def get_tiles_num_per_image(self, image_idx: int):
        return len(self.__images[image_idx]['tiles_ids'])

    def unite_data(self, tiles: list, img_idx) -> np.array:
        """
        Unite tiles to one image by data image inadex
        :param tiles: list of tiles
        :param img_idx: index of images, that has this tiles
        :return: united tiles into one
        """
        if len(tiles) != self.get_tiles_num_per_image(img_idx):
            raise Exception("Tiles number doesn't equal to real tiles size")

        img_size = self.__images[img_idx]['size']

        if len(tiles[0].shape) == 3:
            res = np.zeros((img_size[0], img_size[1], tiles[0].shape[2]), dtype=np.uint8)
            for i, tile in enumerate(tiles):
                [x1, y1], [x2, y2] = self.__tiles[i]['tile']
                res[y1: y2, x1: x2, :] = tile
        else:
            res = np.zeros((img_size[0], img_size[1]), dtype=tiles[0].dtype)
            weights_mask = np.zeros_like(res)
            for i, tile in enumerate(tiles):
                [x1, y1], [x2, y2] = self.__tiles[i]['tile']
                res[y1: y2, x1: x2] += tile
                weights_mask[y1: y2, x1: x2] += np.ones_like(tile, dtype=np.float)
            weights_mask[weights_mask == 0] = 1
            res = res / weights_mask
        return res

    def __getitem__(self, item):
        for tile_idx in self.__images[item]["tiles_ids"]:
            yield super().__getitem__(tile_idx)
