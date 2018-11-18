import unittest

from neural_pipeline.data_producer.data_producer import AbstractDataset


class SampleDataset(AbstractDataset):
    def __init__(self, numbers):
        self.__numbers = numbers

    def __len__(self):
        return len(self.__numbers)

    def __getitem__(self, item):
        return self.__numbers[item]


class DataProducerTest(unittest.TestCase):
    def getitem_test(self):
        numbers = list(range(20))
        dataset1 = SampleDataset(numbers[:4])
        dataset2 = SampleDataset(numbers[4:])


if __name__ == '__main__':
    unittest.main()
