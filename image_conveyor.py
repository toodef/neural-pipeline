from PIL import Image
import requests
from io import BytesIO
from abc import ABCMeta, abstractmethod


class ImageLoader(metaclass=ABCMeta):
    @abstractmethod
    def load(self, path: str):
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
        self.__images_buffer = None
        self.__cur_index = 0

    def load(self, path: str):
        """
        Load image by url
        :param path: path to image
        :return: image object
        """
        self.__image_loader.load(path)

    def __getitem__(self, index):
        self.__swap_buffers()
        return self.__images_buffer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __swap_buffers(self):
        new_buffer = []
        for i in range(self.__images_bucket_size):
            if self.__cur_index + i == len(self.__image_pathes):
                break
            new_buffer.append(self.__image_loader.load(self.__image_pathes[self.__cur_index + i]))
        self.__cur_index += self.__images_bucket_size
        self.__images_buffer = new_buffer
