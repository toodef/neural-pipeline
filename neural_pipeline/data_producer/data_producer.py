import torch
from torch.utils.data import DataLoader
from abc import ABCMeta, abstractmethod


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

        self.__update_datasets_config()

    def need_shuffle_datasets(self, is_need: bool) -> object:
        """
        Is need to shuffle datasets order
        :param is_need: is need? a?
        :return self object
        """
        self.__need_shuffle = is_need
        return self

    def need_pin_memory(self, is_need: bool) -> object:
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
        for i in range(len(self.__datasets) + 1):
            if item < self.__datatsets_idx_space[i]:
                dataset_idx = i - 1
                break
        dataset = self.__datasets[dataset_idx]
        return dataset[item - self.__datatsets_idx_space[dataset_idx]]

    def get_loader(self) -> DataLoader:
        return DataLoader(self, batch_size=self.__batch_size, num_workers=self.__num_workers, shuffle=True,
                          pin_memory=self.__pin_memory)

    def __update_datasets_config(self):
        datasets_len = [len(d) for d in self.__datasets]
        self.__datatsets_idx_space = [0]
        for l in datasets_len:
            self.__datatsets_idx_space.append(l + self.__datatsets_idx_space[-1])
        self.__overall_len = sum(datasets_len)
