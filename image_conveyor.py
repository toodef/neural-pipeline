from PIL import Image
import requests
from io import BytesIO
from abc import ABCMeta, abstractmethod


class ImageLoader(metaclass=ABCMeta):
    @abstractmethod
    def load(self, path: {}):
        """
        Load image
        :param path: path to image
        :return: image object
        """


class UrlLoader(ImageLoader):
    def load(self, image):
        url = image['path']
        try:
            response = requests.get(url, timeout=10)
            if response.ok:
                image['object'] = Image.open(BytesIO(response.content))
            else:
                image['object'] = None
        except Exception:
            image['object'] = None

        return image


class ImageConveyor:
    def __init__(self, image_loader: ImageLoader, pathes: [{}] = None, images_bucket_size: int = 1):
        self.__images_bucket_size = images_bucket_size
        self.__image_loader = image_loader
        self.__image_pathes = pathes
        self.__cur_index = 0
        self.__images_buffers = [None, self.__load_buffer()]

    def load(self, path: str):
        """
        Load image by url
        :param path: path to image
        :return: image object
        """
        self.__image_loader.load(path)

    def __getitem__(self, index):
        if (index - 1) * self.__images_bucket_size >= len(self.__image_pathes):
            raise IndexError
        self.__swap_buffers()
        return self.__images_buffers[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __load_buffer(self):
        new_buffer = []
        for i in range(self.__images_bucket_size):
            if self.__cur_index == len(self.__image_pathes):
                break
            new_buffer.append(self.__image_loader.load(self.__image_pathes[self.__cur_index]))
            self.__cur_index += 1
        return new_buffer

    def __swap_buffers(self):
        new_buffer = self.__load_buffer()
        self.__images_buffers = [self.__images_buffers[1], new_buffer]
