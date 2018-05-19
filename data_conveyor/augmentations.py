import torch
from abc import ABCMeta, abstractmethod
from random import randint
import numpy as np
import cv2
from torchvision.transforms import transforms


class Augmentation(metaclass=ABCMeta):
    def __init__(self, config: {}, aug_name: str):
        self.__aug_name = aug_name
        self._percentage = self._get_config_path(config)['percentage']

    def __call__(self, data):
        """
        Process data
        :param data: data object
        :return: processed data object
        """
        if randint(1, 100) <= self._percentage:
            return self.process(data)
        else:
            return data

    @abstractmethod
    def process(self, data):
        """
        Process data
        :param data: data object
        :return: processed data object
        """

    def _get_config_path(self, config):
        return config[self.__aug_name]

    def get_percetage(self):
        return self._percentage


class HorizontalFlip(Augmentation):
    def __init__(self, config: {}):
        super().__init__(config, 'hflip')

    def process(self, data):
        return cv2.flip(data, 1)


class VerticalFlip(Augmentation):
    def __init__(self, config: {}):
        super().__init__(config, 'vflip')

    def process(self, data):
        return cv2.flip(data, 0)


class GaussNoise(Augmentation):
    def __init__(self, config: {}):
        super().__init__(config, 'gauss_noise')

    def process(self, data):
        row, col, ch = data.shape
        mean = 0
        var = 0.001
        interval = 50
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        # gauss = gauss.reshape(row, col, ch)
        gauss = (gauss - np.min(gauss))
        gauss = gauss / np.max(gauss) * interval
        noisy = data
        mask = data < 255 - interval
        noisy[mask] = data[mask] + gauss[mask]
        return noisy.astype(np.uint8)


class SNPNoise(Augmentation):
    def __init__(self, config: {}):
        super().__init__(config, 'snp_noise')

    def process(self, data):
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(data)
        # Salt mode
        num_salt = np.ceil(amount * data.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in data.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * data.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in data.shape]
        out[coords] = 0
        return out


class Blur(Augmentation):
    def __init__(self, config: {}):
        super().__init__(config, 'blur')

    def process(self, data):
        return cv2.blur(data, (5, 5))


def resize_to_defined(data, size):
    return cv2.resize(data, (size[0], size[1]))


def resize_by_min_edge(data, size):
    min_size_idx = np.argmin(data.shape[0: 2])
    max_size_idx = 1 - min_size_idx
    max_size = size * data.shape[max_size_idx] // data.shape[min_size_idx]
    target_size = (size, max_size) if min_size_idx == 1 else (max_size, size)
    return cv2.resize(data, target_size)


class Resize(Augmentation):
    def __init__(self, config: {}):
        super().__init__(config, 'resize')
        self.__size = self._get_config_path(config)['size']
        self.__resize_fnc = resize_to_defined if type(self.__size) == list and len(
            self.__size) == 2 else resize_by_min_edge
        self._percentage = 100

    def process(self, data):
        return self.__resize_fnc(data, self.__size)


class CentralCrop(Augmentation):
    def __init__(self, config: {}):
        super().__init__(config, 'ccrop')
        size = self._get_config_path(config)['size']
        self.__width, self.__height = size if type(size) == list and len(size) == 2 else [size, size]

    def process(self, data):
        h, w, c = data.shape
        dx, dy = (w - self.__width) // 2, (h - self.__height) // 2
        y1, y2 = dy, dy + self.__height
        x1, x2 = dx, dx + self.__width
        data = data[y1: y2, x1: x2, :]
        return data


class RandomCrop(Augmentation):
    def __init__(self, config: {}):
        super().__init__(config, 'rcrop')
        size = self._get_config_path(config)['size']
        self.__width, self.__height = size if type(size) == list and len(size) == 2 else [size, size]

    def process(self, data):
        h, w, c = data.shape
        dx, dy = randint(0, w - self.__width) if w > self.__width else 0, \
                 randint(0, h - self.__height) if h > self.__height else 0
        y1, y2 = dy, dy + self.__height
        x1, x2 = dx, dx + self.__width
        data = data[y1: y2, x1: x2, :]
        return data


class RandomRotate(Augmentation):
    def __init__(self, config: {}):
        super().__init__(config, 'rrotate')
        self.__interval = self._get_config_path(config)['interval']

    def process(self, data):
        rows, cols = data.shape[:2]
        angle = randint(self.__interval[0], self.__interval[1])
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(data, M, (cols, rows))


class Normalize(Augmentation):
    def __init__(self, config: {}):
        super().__init__(config, 'normalize')
        self.__normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._percentage = 100

    def process(self, data):
        return self.__normalize(data)


class ToPyTorch(Augmentation):
    def __init__(self, config: {}):
        super().__init__(config, 'to_pytorch')
        self._percentage = 100

    def process(self, data):
        if data.dtype == np.uint8:
            return torch.from_numpy(np.moveaxis(data / 255., -1, 0).astype(np.float32))
        else:
            return torch.from_numpy(np.moveaxis(data, -1, 0).astype(np.float32))


augmentations_dict = {'hflip': HorizontalFlip,
                      'vflip': VerticalFlip,
                      'gauss_noise': GaussNoise,
                      'snp_noise': SNPNoise,
                      'blur': Blur,
                      'resize': Resize,
                      'ccrop': CentralCrop,
                      'rcrop': RandomCrop,
                      'rrotate': RandomRotate,
                      'to_pytorch': ToPyTorch,
                      'normalize': Normalize}
