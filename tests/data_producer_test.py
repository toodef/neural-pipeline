import unittest
from random import randint

from torch.utils.data import DataLoader

from neural_pipeline.data_producer.data_producer import AbstractDataset, DataProducer

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
        dataset1 = SampleDataset(numbers[:4])
        self.assertEqual(len(dataset1), 4)
        dataset2 = SampleDataset(numbers[4:])
        self.assertEqual(len(dataset2), 16)

        for i, n in enumerate(numbers):
            if i < len(dataset1):
                self.assertEqual(dataset1[i], n)
            else:
                self.assertEqual(dataset2[i - len(dataset1)], n)

        data_producer = TestDataProducer([dataset1, dataset2])

        for i, n in enumerate(numbers):
            self.assertEqual(data_producer[i], n)

    def test_global_shuffle(self):
        data_producer = DataProducer([list(range(10)), list(range(10, 20))])

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

    def test_shuffle_datasets_order(self):
        data_producer = DataProducer([list(range(10)), list(range(10, 20))]) \
            .shuffle_datasets_order(True)

        prev_data, first_dataset_first = None, None
        for i, data in enumerate(data_producer):
            if first_dataset_first is None:
                first_dataset_first = data < 10
                prev_data = data
                continue

            if i < 10:
                self.assertEqual(data, prev_data + 1)
                self.assertEqual(first_dataset_first, data < 10)

            if i > 9:
                self.assertEqual(first_dataset_first, data >= 10)
                if i > 10:
                    self.assertEqual(data, prev_data + 1)

            prev_data = data

    def test_pin_memory(self):
        data_producer = DataProducer([list(range(1))]).pin_memory(False)
        self.assertFalse(data_producer.get_loader().pin_memory)
        self.assertFalse(data_producer.get_loader([('0_0',)]).pin_memory)

        data_producer.pin_memory(True)
        self.assertTrue(data_producer.get_loader().pin_memory)
        self.assertTrue(data_producer.get_loader([('0_0',)]).pin_memory)

    def test_pass_indices(self):
        data_producer = DataProducer([list(range(10)), list(range(10, 20))])
        loader = data_producer.global_shuffle(True).pass_indices(True).get_loader()

        for i, item in enumerate(loader):
            data, idx = item['data'], item['data_idx'][0]

            self.assertEqual(data, 10 * int(idx[0]) + int(idx[2]))

        indices = list(set([('{}_{}'.format(randint(0, 1), randint(0, 9)),) for _ in range(10)]))

        for data in data_producer.get_loader(indices):
            d = int(data)
            idx = ('{}_{}'.format(int(d > 9), d if d < 10 else d % 10),)
            self.assertIn(idx, indices)
            indices.remove(idx)

        self.assertEqual(len(indices), 0)


if __name__ == '__main__':
    unittest.main()
