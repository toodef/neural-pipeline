import unittest

from neural_pipeline.data_producer.data_producer import AbstractDataset, AbstractDataProducer


class SampleDataset(AbstractDataset):
    def __init__(self, numbers):
        self.__numbers = numbers

    def __len__(self):
        return len(self.__numbers)

    def __getitem__(self, item):
        return self.__numbers[item]


class DataProducer(AbstractDataProducer):
    def __init__(self, datasets: [AbstractDataset]):
        super().__init__(datasets)


class DataProducerTest(unittest.TestCase):
    def test_getitem(self):
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

        data_producer = DataProducer([dataset1, dataset2])

        for i, n in enumerate(numbers):
            self.assertEqual(data_producer[i], n)


if __name__ == '__main__':
    unittest.main()
