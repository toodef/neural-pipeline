import itertools
from random import shuffle

from torch.utils.data import DataLoader
from abc import ABCMeta, abstractmethod

__all__ = ['AbstractDataset', 'DataProducer']


class AbstractDataset(metaclass=ABCMeta):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass


class DataProducer:
    """
    Data Producer. Accumulate one or more datasets and pass it's data by batches for processing.
    This use PyTorch builtin :class:`DataLoader` for increase performance of data delivery.

    :param datasets: list of datasets. Every dataset might be iterable (contans methods ``__getitem__`` and ``__len__``)
    :param batch_size: size of output batch
    :param num_workers: number of processes, that load data from datasets and pass it for output
    """

    def __init__(self, datasets: [AbstractDataset], batch_size: int = 1, num_workers: int = 0):
        self.__datasets = datasets
        self.__batch_size = batch_size
        self.__num_workers = num_workers

        self._shuffle_datasets_order = False
        self._glob_shuffle = False
        self._pin_memory = False

        self._need_pass_indices = False

        self._update_datasets_idx_space()

    def shuffle_datasets_order(self, is_need: bool) -> 'DataProducer':
        """
        Is need to shuffle datasets order. Shuffling performs after every 0 index access

        :param is_need: is need
        :return self object
        """
        self._shuffle_datasets_order = is_need
        return self

    def global_shuffle(self, is_need: bool) -> 'DataProducer':
        """
        Is need global shuffling. If global shuffling enable - batches will compile from random indices of all datasets. In this case datasets order shuffling was ignoring

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
        data = self.__datasets[dataset_idx][data_idx]
        if self._need_pass_indices:
            if not isinstance(data, dict):
                data = {'data': data}
            return dict(data, **{'data_idx': str(dataset_idx) + "_" + str(data_idx)})
        return data

    def __len__(self):
        return self.__overall_len

    def __getitem__(self, item):
        if item == 0 and (not self._glob_shuffle) and self._shuffle_datasets_order:
            self._update_datasets_idx_space()

        dataset_idx = 0
        data_idx = item
        for i in range(len(self.__datasets)):
            if item > self.__datatsets_idx_space[i]:
                dataset_idx = i + 1
                data_idx = item - self.__datatsets_idx_space[i] - 1

        return self.get_data(dataset_idx, data_idx)

    def get_loader(self, indices: [str] = None) -> DataLoader:
        """
        Get PyTorch :class:`DataLoader` object, that aggregate :class:`DataProducer`.
        If ``indices`` is specified - DataLoader wil output data only by this indices. In this case indices will not passed.

        :param indices: list of indices. Each item of list is a string in format '{}_{}'.format(dataset_idx, data_idx)
        :return: :class:`DataLoader` object
        """
        if indices is not None:
            return self._get_loader_by_indices(indices)
        return DataLoader(self, batch_size=self.__batch_size, num_workers=self.__num_workers,
                          shuffle=self._glob_shuffle, pin_memory=self._pin_memory)

    def _get_loader_by_indices(self, indices: [str]) -> DataLoader:
        """
        Get loader, that produce data only by specified indices

        :param indices: required indices
        :return: :class:`DataLoader` object
        """
        return DataLoader(_ByIndices(self.__datasets, indices), batch_size=self.__batch_size, num_workers=self.__num_workers,
                          shuffle=self._glob_shuffle, pin_memory=self._pin_memory)

    def _update_datasets_idx_space(self) -> None:
        """
        Update idx space of datasets. Idx space used for correct mapping global idx to corresponding dataset data index
        """
        if self._shuffle_datasets_order:
            shuffle(self.__datasets)

        datasets_len = [len(d) for d in self.__datasets]
        self.__overall_len = sum(datasets_len)
        self.__datatsets_idx_space = []
        cur_len = 0
        for dataset_len in datasets_len:
            self.__datatsets_idx_space.append(dataset_len + cur_len - 1)
            cur_len += dataset_len


class _ByIndices(DataProducer):
    def __init__(self, datasets: [AbstractDataset], indices: []):
        super().__init__(datasets)
        self.shuffle_datasets_order(False)
        self.indices = list(itertools.chain(*indices))

    def __getitem__(self, item):
        dataset_idx, data_idx = self.indices[item].split('_')
        return self.get_data(int(dataset_idx), int(data_idx))

    def __len__(self):
        return len(self.indices)
