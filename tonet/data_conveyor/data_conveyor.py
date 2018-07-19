from random import randint

import cv2


from .augmentations import augmentations_dict


class Dataset:
    def __init__(self, config: {}, pathes: []):
        self.__pathes = pathes
        percentage = config['images_percentage'] if 'images_percentage' in config else 100
        self.__cell_size = 100 / percentage
        self.__data_num = len(self.__pathes['data']) * percentage // 100
        self.__percentage = percentage

        self.load_augmentations(config)

        self.__augmentations_percentage = config['augmentations_percentage'] if 'augmentations_percentage' in config else None

    def load_augmentations(self, config: []):
        """
        Load augmentations by config
        :param config: list of augmentations config
        :return:
        """
        before_augmentations_config = config['before_augmentations']
        self.__before_augmentations = [augmentations_dict[next(iter(aug))](aug) for aug in before_augmentations_config]
        if 'augmentations' in config:
            self.__augmentations = [augmentations_dict[next(iter(aug))](aug) for aug in config['augmentations']]
        else:
            self.__augmentations = []
        after_augmentations_config = config['after_augmentations']
        self.__after_augmentations = [augmentations_dict[next(iter(aug))](aug) for aug in after_augmentations_config]

    def __getitem__(self, item):
        def augmentate(image):
            for aug in self.__before_augmentations:
                image = aug(image)
            if self.__augmentations_percentage is not None and randint(1, 100) <= self.__augmentations_percentage:
                for aug in self.__augmentations:
                    image = aug(image)
            for aug in self.__after_augmentations:
                image = aug(image)
            return image

        item = randint(1, self.__cell_size) + int(item * self.__cell_size) - 1 if self.__percentage < 100 else item

        if 'target' in self.__pathes['data'][item]:
            return {'data': augmentate(cv2.imread(self.__pathes['data'][item]['path'])),
                    'target': self.__pathes['data'][item]['target']}
        else:
            return augmentate(cv2.imread(self.__pathes['data'][item]['path']))

    def __len__(self):
        return self.__data_num
