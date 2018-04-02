import unittest

from image_conveyor import ImageConveyor, ImageLoader


class ImageConveyorTest(unittest.TestCase):
    class TestImageLoader(ImageLoader):
        def load(self, path: {}):
            path['additional_data'] = path['path'] + "add"
            return path

    def test_get_data_test(self):
        PATHES_NUM = 23
        BUCKET_SIZE = 7
        pathes = [{'path': str(path), 'some_data': path * 31} for path in range(PATHES_NUM)]
        getting_pathes = []
        with ImageConveyor(self.TestImageLoader(), pathes, BUCKET_SIZE) as conveyor:
            for images in conveyor:
                for img in images:
                    getting_pathes.append(img)

        self.assertEqual(len(pathes), len(getting_pathes))


if __name__ == '__main__':
    unittest.main()
