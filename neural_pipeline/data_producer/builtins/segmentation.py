from abc import abstractmethod, ABCMeta

import cv2
import numpy as np

from tonet.neural_pipeline.data_producer.data_producer import AbstractDataset


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


class Dataset(AbstractDataset):
    """
    Classic dataset, that just load images from disk
    """

    def __init__(self, pathes: [], mask_interpreter: AbstractMaskInterpreter):
        """
        :param pathes: dataset data: pairs of pathes (image and mask)
        :param mask_interpreter: concrette mask interpreter
        """
        super().__init__()

        self.__pathes = pathes
        self.__mask_interpreter = mask_interpreter

    def __len__(self):
        return len(self.__pathes)

    def __getitem__(self, item):
        image = cv2.imread(self.__pathes[item]['image'], -1)
        return self.__mask_interpreter(image.shape, self.__pathes[item]['mask'])


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

    def __init__(self, pathes: [], mask_interpreter: AbstractMaskInterpreter, tile_size: list, img_original_size: list = None):
        """
        :param pathes: dataset date. Like pathes, targets, labels
        :param mask_interpreter: concrette mask interpreter
        :param tile_size: size of tiles
        :param img_original_size: images size (optional)
        """

        super().__init__()

        self.__pathes = pathes
        self.__mask_interpreter = mask_interpreter

        self.__tiles = []
        self.__images = []
        if img_original_size is None:
            for i, data in enumerate(self.__pathes):
                img = cv2.imread(data['image'])
                img_size = [img.shape[0], img.shape[1]]
                cur_tiles = []
                for tile in self.__get_image_tiles_by_size([img_size[1], img_size[0]], tile_size):
                    cur_tiles.append({"tile": tile, "img_id": i})
                img_info = {"path": data['path'], "tiles_ids": list(range(len(self.__tiles), len(self.__tiles) + len(cur_tiles))),
                            "size": img_size}
                if "target" in data:
                    img_info["target"] = data['target']
                self.__images.append(img_info)
                self.__tiles.extend(cur_tiles)
        else:
            one_image_tiles = self.__get_image_tiles_by_size(img_original_size, tile_size)
            for i, data in enumerate(self.__pathes):
                cur_tiles = []
                for tile in one_image_tiles:
                    cur_tiles.append({"tile": tile, "img_id": i})
                img_info = {"path": data['path'], "tiles_ids": list(range(len(self.__tiles), len(self.__tiles) + len(cur_tiles))),
                            "size": img_original_size}
                if "target" in data:
                    img_info["target"] = data['target']
                self.__images.append(img_info)
                self.__tiles.extend(cur_tiles)

    def __get_image_tiles_by_size(self, img_original_size, tile_size: list) -> []:
        """
        Calculate tiles by image size
        :param img_original_size: image size
        :param tile_size: tile size
        :return: list of tiles
        """
        return self.MeshGenerator([[0, 0], img_original_size], tile_size).generate_cells()

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
        return {'data': res['data'][y1: y2, x1: x2, :], 'mask': res['mask'][y1: y2, x1: x2]} if 'mask' in res else {
            'data': res['data'][y1: y2, x1: x2, :]}

    def __len__(self):
        return len(self.__tiles)

    def _remove_item_by_idx(self, idx):
        del self.__tiles[idx]

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
