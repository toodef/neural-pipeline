import os
import numpy as np
from abc import ABCMeta, abstractmethod

__all__ = ['DatasetException', 'get_root_by_env', 'AbstractIndexedDataset', 'AbstractDataset', 'IndexedDataset', 'BasicDataset']


class AbstractDataset(metaclass=ABCMeta):
    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass


class DatasetException(Exception):
    def __init__(self, msg: str):
        self._msg = msg

    def __str__(self) -> str:
        return self._msg


def get_root_by_env(env_name: str) -> str:
    """
    Get dataset root by environment variable

    :param env_name: environment variable name
    :return: path to dataset root
    """
    if env_name not in os.environ:
        raise DatasetException("Can't get dataset root. Please define '" + env_name + "' environment variable")
    return os.environ[env_name]


class AbstractIndexedDataset(metaclass=ABCMeta):
    """
    Interface for work with indices in datasets
    """

    def __init__(self):
        self._indices = None
        self._use_indices = False

    def set_indices(self, indices: [np.uint]) -> 'AbstractIndexedDataset':
        self._indices = indices
        self._use_indices = True
        return self

    def get_indices(self) -> [np.uint]:
        return self._indices

    def use_indices(self, need_use: bool = True) -> 'AbstractIndexedDataset':
        self._use_indices = need_use
        return self

    def remove_indices(self) -> 'AbstractIndexedDataset':
        self._indices = None
        self.use_indices(False)
        return self

    def load_indices(self, path: str) -> 'AbstractIndexedDataset':
        self.set_indices(np.load(path))
        return self

    def flush_indices(self, path: str) -> 'AbstractIndexedDataset':
        if self._indices is None:
            raise DatasetException
        np.save(path, self._indices)
        return self


class IndexedDataset(AbstractIndexedDataset, AbstractDataset, metaclass=ABCMeta):
    pass


class BasicDataset(IndexedDataset):
    """
    The standard dataset basic class.

    Basic dataset get array of items and works with it. Array of items is just an array of shape [N, ?]
    """

    def __init__(self, items: list):
        super().__init__()
        self._items = items

    def get_items(self) -> list:
        """
        Get array of items

        :return: array of indices
        """
        return self._items

    @abstractmethod
    def _interpret_item(self, item) -> any:
        """
        Interpret one item from dataset. This method get index of item and returns interpreted data? that will be passed from dataset

        Args:
            item: item of items array

        Returns:
            One item, that
        """

    def remove_unused_data(self):
        self._items = [self._items[idx] for idx in self._indices]
        self._use_indices = False

    def __len__(self):
        if self._use_indices:
            return len(self._indices)
        return len(self._items)

    def __getitem__(self, idx):
        if self._use_indices:
            return self._interpret_item(self._items[self._indices[idx]])
        else:
            return self._interpret_item(self._items[idx])
