from PIL import Image
import requests
from io import BytesIO
from abc import ABCMeta, abstractmethod
from multiprocessing import Pool


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


def load_image(info: [int, ImageLoader, {}]):
    if info[0] == len(info[2]):
        return None
    return info[1].load(info[2])


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
        # return self.__load_buffer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __load_buffer(self):
        threads_data = [[idx, self.__image_loader, self.__image_pathes[idx]] for idx in
                        range(self.__cur_index, self.__cur_index + self.__images_bucket_size) if
                        idx < len(self.__image_pathes)]
        if len(threads_data) == 0:
            return []
        if len(threads_data) == 1:
            return [load_image(threads_data[0])]
        pool = Pool(len(threads_data))
        new_buffer = pool.map(load_image, threads_data)
        self.__cur_index += self.__images_bucket_size
        return new_buffer

    def __swap_buffers(self):
        del self.__images_buffers[0]
        self.__images_buffers.append(self.__load_buffer())
