import unittest
from random import randint

from torch.utils.data import DataLoader

from piepline import DataProducer
from piepline.data_producer.datasets import AbstractDataset

__all__ = ['DataProducerTest']


class SampleDataset(AbstractDataset):
    def __init__(self, numbers):
        self.__numbers = numbers

    def __len__(self):
        return len(self.__numbers)

    def __getitem__(self, item):
        return self.__numbers[item]


class TestDataProducer(DataProducer):
    def __init__(self, datasets: [AbstractDataset]):
        super().__init__(datasets)


class DataProducerTest(unittest.TestCase):
    def test_simple_getitem(self):
        numbers = list(range(20))
        dataset = SampleDataset(numbers)
        self.assertEqual(len(dataset), 20)

        for i, n in enumerate(numbers):
            self.assertEqual(dataset[i], n)

        data_producer = TestDataProducer(dataset)

        for i, n in enumerate(numbers):
            self.assertEqual(data_producer[i], n)

    def test_global_shuffle(self):
        data_producer = DataProducer(list(range(10)))

        prev_data = None
        for data in data_producer:
            if prev_data is None:
                prev_data = data
                continue
            self.assertEqual(data, prev_data + 1)
            prev_data = data

        data_producer.global_shuffle(True)
        prev_data = None
        shuffled, non_shuffled = 1, 1
        for data in data_producer:
            if prev_data is None:
                prev_data = data
                continue
            if prev_data + 1 == data:
                non_shuffled += 1
            else:
                shuffled += 1
            prev_data = data

        self.assertEqual(non_shuffled, len(data_producer))

        prev_data = None
        shuffled, non_shuffled = 0, 0
        loader = data_producer.get_loader()
        for data in loader:
            if prev_data is None:
                prev_data = data
                continue
            if prev_data + 1 == data:
                non_shuffled += 1
            else:
                shuffled += 1
            prev_data = data

        self.assertGreater(shuffled, non_shuffled)

    def test_get_loader(self):
        data_producer = DataProducer([list(range(1))])
        self.assertIs(type(data_producer.get_loader()), DataLoader)
        self.assertIs(type(data_producer.get_loader([('0_0',)])), DataLoader)

    def test_pin_memory(self):
        data_producer = DataProducer([list(range(1))]).pin_memory(False)
        self.assertFalse(data_producer.get_loader().pin_memory)
        self.assertFalse(data_producer.get_loader([('0_0',)]).pin_memory)

        data_producer.pin_memory(True)
        self.assertTrue(data_producer.get_loader().pin_memory)
        self.assertTrue(data_producer.get_loader([('0_0',)]).pin_memory)

    def test_pass_indices(self):
        data_producer = DataProducer(list(range(10)))
        loader = data_producer.global_shuffle(True).pass_indices(True).get_loader()

        for i, item in enumerate(loader):
            data, idx = item['data'], item['data_idx'][0]
            self.assertEqual(data, int(idx))

        indices = list([str(i) for i in range(10)])

        for data in data_producer.get_loader(indices):
            d = int(data)
            idx = ('{}'.format(d if d < 10 else d % 10))
            self.assertIn(idx, indices)
            indices.remove(idx)

        self.assertEqual(len(indices), 0)


if __name__ == '__main__':
    unittest.main()
