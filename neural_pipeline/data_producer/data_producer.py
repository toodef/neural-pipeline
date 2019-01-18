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
    def __init__(self, datasets: [AbstractDataset], batch_size: int = 1, num_workers: int = 0):
        self.__datasets = datasets
        self.__batch_size = batch_size
        self.__num_workers = num_workers

        self.__need_shuffle = False
        self.__pin_memory = False

        self.__update_datasets_idx_space()

    def need_shuffle_datasets(self, is_need: bool) -> object:
        """
        Is need to shuffle datasets order
        :param is_need: is need? a?
        :return self object
        """
        self.__need_shuffle = is_need
        return self

    def need_pin_memory(self, is_need: bool) -> 'DataProducer':
        """
        Is need to pin memory on loading
        :param is_need: is need
        :return: self object
        """
        self.__pin_memory = is_need
        return self

    def __len__(self):
        return self.__overall_len

    def __getitem__(self, item):
        dataset_idx = 0
        data_idx = item
        for i in range(len(self.__datasets)):
            if item > self.__datatsets_idx_space[i]:
                dataset_idx = i + 1
                data_idx = item - self.__datatsets_idx_space[i] - 1

        dataset = self.__datasets[dataset_idx]
        return dataset[data_idx]

    def get_loader(self) -> DataLoader:
        return DataLoader(self, batch_size=self.__batch_size, num_workers=self.__num_workers, shuffle=True,
                          pin_memory=self.__pin_memory)

    def __update_datasets_idx_space(self):
        datasets_len = [len(d) for d in self.__datasets]
        self.__overall_len = sum(datasets_len)
        self.__datatsets_idx_space = []
        cur_len = 0
        for dataset_len in datasets_len:
            self.__datatsets_idx_space.append(dataset_len + cur_len - 1)
            cur_len += dataset_len
