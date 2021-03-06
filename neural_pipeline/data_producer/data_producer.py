import itertools

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from abc import ABCMeta, abstractmethod
import numpy as np


__all__ = ['AbstractDataset', 'DataProducer', 'BasicDataset']


class AbstractDataset(metaclass=ABCMeta):
    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass


class IndexesDataset(AbstractDataset):

class BasicDataset(AbstractDataset, metaclass=ABCMeta):
    """
    The standard dataset basic class.

    Basic dataset get array of items and works with it. Array of items is just an array of shape [N, ?]
    """
    def __init__(self, items: list):
        self._items = items
        self._indices = None

    @abstractmethod
    def _interpret_item(self, item) -> any:
        """
        Interpret one item from dataset. This method get index of item and returns interpreted data? that will be passed from dataset

        Args:
            item: item of items array

        Returns:
            One item, that
        """

    def get_items(self) -> list:
        """
        Get array of items

        :return: array of indices
        """
        return self._items

    def set_indices(self, indices: [int], remove_unused: bool = False) -> 'AbstractDataset':
        if remove_unused:
            self._items = [self._items[idx] for idx in indices]
            self._indices = None
        else:
            self._indices = indices
        return self

    def get_indices(self) -> list:
        return self._indices

    def load_indices(self, path: str, remove_unused: bool = False) -> 'AbstractDataset':
        self.set_indices(np.load(path), remove_unused)
        return self

    def flush_indices(self, path: str) -> 'AbstractDataset':
        np.save(path, self._indices)
        return self

    def __getitem__(self, idx):
        if self._indices is None:
            return self._interpret_item(self._items[idx])
        else:
            return self._interpret_item(self._items[self._indices[idx]])

    def __len__(self):
        if self._indices is None:
            return len(self._items)
        else:
            return len(self._indices)


class DataProducer:
    """
    Data Producer. Accumulate one or more datasets and pass it's data by batches for processing.
    This use PyTorch builtin :class:`DataLoader` for increase performance of data delivery.

    :param datasets: list of datasets. Every dataset might be iterable (contans methods ``__getitem__`` and ``__len__``)
    :param batch_size: size of output batch
    :param num_workers: number of processes, that load data from datasets and pass it for output
    """

    def __init__(self, dataset: [AbstractDataset], batch_size: int = 1, num_workers: int = 0):
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers

        self._glob_shuffle = True
        self._pin_memory = False
        self._collate_fn = default_collate
        self._drop_last = False

        self._need_pass_indices = False

    def drop_last(self, need_drop: bool) -> 'DataProducer':
        self._drop_last = need_drop
        return self

    def global_shuffle(self, is_need: bool) -> 'DataProducer':
        """
        Is need global shuffling. If global shuffling enable - batches will compile from random indices.

        :param is_need: is need global shuffling
        :return: self object
        """
        self._glob_shuffle = is_need
        return self

    def pin_memory(self, is_need: bool) -> 'DataProducer':
        """
        Is need to pin memory on loading. Pinning memory was increase data loading performance (especially when data loads to GPU) but incompatible with swap

        :param is_need: is need
        :return: self object
        """
        self._pin_memory = is_need
        return self

    def pass_indices(self, need_pass: bool) -> 'DataProducer':
        """
        Pass indices of data in every batch. By default disabled

        :param need_pass: is need to pass indices
        """
        self._need_pass_indices = need_pass
        return self

    def _is_passed_indices(self) -> bool:
        """
        Internal method for know if :class:`DataProducer` passed indices

        :return: is passed
        """
        return self._need_pass_indices

    def get_data(self, dataset_idx: int, data_idx: int) -> object:
        """
        Get single data by dataset idx and data_idx

        :param dataset_idx: index of dataset
        :param data_idx: index of data in this dataset
        :return: dataset output
        """
        data = self._dataset[data_idx]
        if self._need_pass_indices:
            if not isinstance(data, dict):
                data = {'data': data}
            return dict(data, **{'data_idx': str(dataset_idx) + "_" + str(data_idx)})
        return data

    def set_collate_func(self, func: callable) -> 'DataProducer':
        self._collate_fn = func
        return self

    def get_loader(self, indices: [str] = None) -> DataLoader:
        """
        Get PyTorch :class:`DataLoader` object, that aggregate :class:`DataProducer`.
        If ``indices`` is specified - DataLoader will output data only by this indices. In this case indices will not passed.

        :param indices: list of indices. Each item of list is a string in format '{}_{}'.format(dataset_idx, data_idx)
        :return: :class:`DataLoader` object
        """
        if indices is not None:
            return self._get_loader_by_indices(indices)
        return DataLoader(self._dataset, batch_size=self._batch_size, num_workers=self._num_workers,
                          shuffle=self._glob_shuffle, pin_memory=self._pin_memory, collate_fn=self._collate_fn,
                          drop_last=self._drop_last)

    def _get_loader_by_indices(self, indices: [str]) -> DataLoader:
        """
        Get loader, that produce data only by specified indices

        :param indices: required indices
        :return: :class:`DataLoader` object
        """
        return DataLoader(_ByIndices(self._dataset, indices), batch_size=self._batch_size, num_workers=self._num_workers,
                          shuffle=self._glob_shuffle, pin_memory=self._pin_memory, collate_fn=self._collate_fn,
                          drop_last=self._drop_last)


class _ByIndices(DataProducer):
    def __init__(self, dataset:AbstractDataset, indices: list):
        super().__init__(dataset)
        self.indices = list(itertools.chain(*indices))

    def __getitem__(self, item):
        dataset_idx, data_idx = self.indices[item].split('_')
        return self.get_data(int(dataset_idx), int(data_idx))

    def __len__(self):
        return len(self.indices)
