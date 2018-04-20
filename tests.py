import unittest

from data_conveyor import DataConveyor, DataLoader


class ImageConveyorTest(unittest.TestCase):
    class TestDataLoader(DataLoader):
        def _load(self, path: {}):
            path['additional_data'] = path['path'] + "add"
            return path

    def test_get_data(self):
        PATHES_NUM = 23
        BUCKET_SIZE = 7
        pathes = [{'path': str(path), 'some_data': path * 31} for path in range(PATHES_NUM)]
        getting_pathes = []
        with DataConveyor(self.TestDataLoader(), pathes, BUCKET_SIZE) as conveyor:
            for images in conveyor:
                for img in images:
                    getting_pathes.append(img)

        self.assertEqual(len(pathes), len(getting_pathes))

        getting_pathes = []
        with DataConveyor(self.TestDataLoader(), pathes, BUCKET_SIZE, get_images_num=100) as conveyor:
            for images in conveyor:
                for img in images:
                    getting_pathes.append(img)

        self.assertEqual(100, len(getting_pathes))

        getting_pathes = []
        with DataConveyor(self.TestDataLoader(), pathes, BUCKET_SIZE, get_images_num=100) as conveyor:
            conveyor.set_processes_num(3)
            for images in conveyor:
                for img in images:
                    getting_pathes.append(img)

        self.assertEqual(100, len(getting_pathes))


if __name__ == '__main__':
    unittest.main()
