import cv2
import numpy as np
import requests
from abc import ABCMeta, abstractmethod
from multiprocessing import Pool
from threading import Thread


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
        try:
            response = requests.get(image['path'], timeout=100)
            if response.ok:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image['object'] = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                image['object'] = None
        except:
            image['object'] = None

        return image


def load_image(info: [ImageLoader, {}]):
    return UrlLoader().load(info[1])


class ImageConveyor:
    def __init__(self, image_loader: ImageLoader, pathes: [{}] = None, images_bucket_size: int = 1):
        self.__images_bucket_size = images_bucket_size
        self.__image_loader = image_loader
        self.__image_pathes = pathes
        self.__cur_index = 0
        self.__buffer_is_ready = False
        self.__buffer_load_thread = None
        self.__images_buffers = [None, None]
        self.__swap_buffers()

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
        if not self.__buffer_is_ready:
            self.__buffer_load_thread.join()
        self.__swap_buffers()
        return self.__images_buffers[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __load_buffer(self):
        threads_data = [[self.__image_loader, self.__image_pathes[idx]] for idx in
                        range(self.__cur_index, (self.__cur_index + self.__images_bucket_size)
                        if (self.__cur_index + self.__images_bucket_size) < len(self.__image_pathes)
                        else len(self.__image_pathes))]
        if len(threads_data) == 0:
            return []
        if len(threads_data) == 1:
            self.__cur_index += 1
            return [load_image(threads_data[0])]
        pool = Pool(100)
        try:
            new_buffer = pool.map(load_image, threads_data)
            pool.close()
        except:
            print(len(threads_data))
            print(threads_data)
            self.__cur_index += self.__images_bucket_size
            return []
        self.__cur_index += self.__images_bucket_size
        return new_buffer

    def __swap_buffers(self):
        def process():
            self.__images_buffers.append(self.__load_buffer())
            self.__buffer_is_ready = True

        del self.__images_buffers[0]
        self.__buffer_is_ready = False
        self.__buffer_load_thread = Thread(target=process)
        self.__buffer_load_thread.start()
