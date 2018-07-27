import json
import os
from random import randint, shuffle

import cv2
import numpy as np

from data_conveyor.augmentations import augmentations_dict, Augmentation
from neuro_studio.PySide2Wrapper.PySide2Wrapper import ImageLayout, CheckBox, LineEdit, ModalWindow, ABCMeta, \
    abstractmethod, Button


class AbstractAugmentationUi(ModalWindow, metaclass=ABCMeta):
    def __init__(self, aug_name, dataset_path: str, project_dir_path: str, previous_augmentations: [] = None):
        super().__init__(aug_name)

        with open(dataset_path, 'r') as dataset:
            dataset_conf = json.load(dataset)

        self.__pathes = [{'path': os.path.join(project_dir_path, p['path']), 'target': p['target']} if type(p['target']) == list else {'path': os.path.join(project_dir_path, p['path'])} for p in dataset_conf['data']]
        shuffle(self.__pathes)

        self.__previous_augs = previous_augmentations

        def read_next():
            self.__cur_img_idx += 1
            self.__read_image(self.__cur_img_idx, True)
            original_img_layout.set_image_from_data(cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB), self._image.shape[1], self._image.shape[0], self._image.shape[1] * self._image.shape[2])

        def read_prev():
            self.__cur_img_idx -= 1
            self.__read_image(self.__cur_img_idx, True)
            original_img_layout.set_image_from_data(cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB), self._image.shape[1], self._image.shape[0], self._image.shape[1] * self._image.shape[2])

        self.start_horizontal()
        self.add_widget(Button("Update", is_tool_button=True).set_on_click_callback(self.update))
        self.add_widget(Button("<--", is_tool_button=True).set_on_click_callback(read_prev))
        self.add_widget(Button("-->", is_tool_button=True).set_on_click_callback(read_next))
        self.cancel()
        self._percentage = self.add_widget(LineEdit().add_label("Percentage", 'left'))
        self.start_horizontal()
        original_img_layout = self.add_widget(ImageLayout())
        self._image_layout = self.add_widget(ImageLayout())
        self.cancel()

        self.__cur_img_idx = 0
        self.__read_image(0, with_augmentations=False)
        original_img_layout.set_image_from_data(cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB), self._image.shape[1], self._image.shape[0], self._image.shape[1] * self._image.shape[2])

    def update(self, with_augmentations=True):
        if with_augmentations:
            aug = self.get_augmentation_instance()
            if self._mask is not None:
                if aug.is_changed_geometry():
                    img, mask = self.get_augmentation_instance()(self._image, self._mask)
                    _, cntrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                else:
                    img = self.get_augmentation_instance()(self._image)
                    _, cntrs, _ = cv2.findContours(self._mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                img = cv2.drawContours(img.copy(), cntrs, -1, (0, 0, 255), 1)
            else:
                img = self.get_augmentation_instance()(self._image)
        else:
            img = self._image
        self._image_layout.set_image_from_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), img.shape[1], img.shape[0],
                                               img.shape[1] * img.shape[2])

    def __read_image(self, image_id: int, with_augmentations=True):
        path = self.__pathes[image_id]

        self._image = cv2.imread(path['path'])

        if 'target' in path:
            cntrs = path['target']
            new_cntrs = np.array([np.array([[p] for p in c]) for c in cntrs])
            self._mask = cv2.drawContours(np.zeros((self._image.shape[0], self._image.shape[1]), dtype=np.uint8), new_cntrs, -1, 255, -1)
        else:
            self._mask = None

        if self.__previous_augs is not None:
            for aug in self.__previous_augs:
                if self._mask is not None and aug.is_changed_geometry():
                    self._image, self._mask = aug(self._image, self._mask)
                else:
                    self._image = aug(self._image)

        self.update(with_augmentations)

    @abstractmethod
    def get_config(self) -> {}:
        pass

    @abstractmethod
    def _init_by_config(self, config: {}):
        pass

    def init_by_config(self, config: {}):
        cur_config = config[GaussNoiseUi._get_name_by_config(config)]
        self._percentage.set_value(str(cur_config['percentage']))
        self._init_by_config(cur_config)

    def show(self) -> Augmentation:
        super().show()
        return self.get_augmentation_instance()

    def get_augmentation_instance(self) -> Augmentation:
        config = self.get_config()
        return augmentations_dict[AbstractAugmentationUi._get_name_by_config(config)](config)

    @staticmethod
    def _get_name_by_config(config: {}) -> str:
        return [k for k, v in config.items()][0]


class ResizeUi(AbstractAugmentationUi):
    def __init__(self, dataset_path: str, project_dir_path: str, previous_augmentations: [] = None):
        super().__init__("Resize", dataset_path, project_dir_path, previous_augmentations)

        self.__check = self.add_widget(CheckBox("By minimum edge").set_value(True))

        self.start_horizontal()
        self.insert_text_label("Size: ")
        self.__size_h = self.add_widget(LineEdit())
        self.insert_text_label("X")
        self.__size_w = self.add_widget(LineEdit()).set_enabled(False)
        self.__check.add_clicked_callback(lambda: self.__size_w.set_enabled(not self.__check.get_value()))
        self.cancel()

    def _init_by_config(self, config: {}):
        size = config['size']
        if type(size) == int:
            self.__size_h.set_value(str(size))
        elif type(size) == list and len(size) == 2:
            self.__size_h.set_value(str(size[0]))
            self.__size_w.set_value(str(size[1]))
            self.__check.set_value(True)

    def get_config(self):
        return {
            'resize': {'percentage': 100, 'size': int(self.__size_h.get_value()) if self.__check.get_value() else
            [int(self.__size_h.get_value()), int(self.__size_w.get_value())]}}


class HorizontalFlipUi(AbstractAugmentationUi):
    def __init__(self, dataset_path: str, project_dir_path: str, previous_augmentations: [] = None):
        super().__init__("Horizontal Flip", dataset_path, project_dir_path, previous_augmentations)

    def _init_by_config(self, config: {}):
        pass

    def get_config(self):
        return {'hflip': {'percentage': 100}}


class VerticalFlipUi(AbstractAugmentationUi):
    def __init__(self, dataset_path: str, project_dir_path: str, previous_augmentations: [] = None):
        super().__init__("Vertical Flip", dataset_path, project_dir_path, previous_augmentations)

    def _init_by_config(self, config: {}):
        pass

    def get_config(self):
        return {'vflip': {'percentage': 100}}


class GaussNoiseUi(AbstractAugmentationUi):
    def __init__(self, dataset_path: str, project_dir_path: str, previous_augmentations: [] = None):
        super().__init__("Gauss Noise", dataset_path, project_dir_path, previous_augmentations)

        self.start_horizontal()
        self.__mean = self.add_widget(LineEdit().add_label("Mean", 'left').set_value_changed_callback(self.update))
        self.__var = self.add_widget(LineEdit().add_label("Var", 'left').set_value_changed_callback(self.update))
        self.__interval = self.add_widget(
            LineEdit().add_label("Interval", 'left').set_value_changed_callback(self.update))
        self.cancel()

    def _init_by_config(self, config: {}):
        self.__mean.set_value(str(config['mean']))
        self.__var.set_value(str(config['var']))
        self.__interval.set_value(str(config['interval']))

    def get_config(self):
        return {'gauss_noise': {'percentage': 100,
                                'mean': float(self.__mean.get_value()),
                                'var': float(self.__var.get_value()),
                                'interval': int(self.__interval.get_value())}}


class SNPNoiseUi(AbstractAugmentationUi):
    def __init__(self, dataset_path: str, project_dir_path: str, previous_augmentations: [] = None):
        super().__init__("Salt&Paper Noise", dataset_path, project_dir_path, previous_augmentations)

        self.start_horizontal()
        self.__s_vs_p = self.add_widget(
            LineEdit().add_label("Salt vs paper", 'left').set_value_changed_callback(self.update))
        self.__amount = self.add_widget(LineEdit().add_label("Amount", 'left').set_value_changed_callback(self.update))
        self.cancel()

    def _init_by_config(self, config: {}):
        self.__s_vs_p.set_value(str(config['s_vs_p']))
        self.__amount.set_value(str(config['amount']))

    def get_config(self):
        return {'snp_noise': {'percentage': 100,
                              's_vs_p': float(self.__s_vs_p.get_value()),
                              'amount': float(self.__amount.get_value())}}


class BlurUi(AbstractAugmentationUi):
    def __init__(self, dataset_path: str, project_dir_path: str, previous_augmentations: [] = None):
        super().__init__("Blur", dataset_path, project_dir_path, previous_augmentations)

        self.start_horizontal()
        self.insert_text_label("Kernel size: ")
        self.__ksize_x = self.add_widget(LineEdit().set_value_changed_callback(self.update))
        self.insert_text_label("X")
        self.__ksize_y = self.add_widget(LineEdit().set_value_changed_callback(self.update))
        self.cancel()

    def _init_by_config(self, config: {}):
        self.__ksize_x.set_value(str(config['ksize'][0]))
        self.__ksize_y.set_value(str(config['ksize'][1]))

    def get_config(self):
        return {'blur': {'percentage': 100,
                         'ksize': (int(self.__ksize_x.get_value()),
                                   int(self.__ksize_y.get_value()))}}


class RandomRotateUi(AbstractAugmentationUi):
    def __init__(self, dataset_path: str, project_dir_path: str, previous_augmentations: [] = None):
        super().__init__("Random Rotate", dataset_path, project_dir_path, previous_augmentations)

        self.start_horizontal()
        self.insert_text_label("Interval, [deg]: ")
        self.__interval_from = self.add_widget(LineEdit().set_value_changed_callback(self.update))
        self.insert_text_label("X")
        self.__interval_to = self.add_widget(LineEdit().set_value_changed_callback(self.update))
        self.cancel()

    def _init_by_config(self, config: {}):
        self.__interval_from.set_value(str(config['interval'][0]))
        self.__interval_to.set_value(str(config['interval'][1]))

    def get_config(self):
        return {'rrotate': {'percentage': 100,
                            'interval': [int(self.__interval_from.get_value()), int(self.__interval_to.get_value())]}}


class CentralCropUi(AbstractAugmentationUi):
    def __init__(self, dataset_path: str, project_dir_path: str, previous_augmentations: [] = None):
        super().__init__("Central Crop", dataset_path, project_dir_path, previous_augmentations)

        self.__check = self.add_widget(CheckBox("Quad").set_value(True))

        self.start_horizontal()
        self.insert_text_label("Size: ")
        self.__size_h = self.add_widget(LineEdit())
        self.insert_text_label("X")
        self.__size_w = self.add_widget(LineEdit()).set_enabled(False)
        self.cancel()

        self.__check.add_clicked_callback(lambda: self.__size_w.set_enabled(not self.__check.get_value()))

    def _init_by_config(self, config: {}):
        size = config['size']
        if type(size) == int:
            self.__size_h.set_value(str(size))
        elif type(size) == list and len(size) == 2:
            self.__size_h.set_value(str(size[0]))
            self.__size_w.set_value(str(size[1]))
            self.__check.set_value(True)

    def get_config(self):
        return {'ccrop': {'percentage': 100,
                          'size': int(self.__size_h.get_value()) if self.__check.get_value() else
                          [int(self.__size_h.get_value()), int(self.__size_w.get_value())]}}


class RandomCropUi(AbstractAugmentationUi):
    def __init__(self, dataset_path: str, project_dir_path: str, previous_augmentations: [] = None):
        super().__init__("Random Crop", dataset_path, project_dir_path, previous_augmentations)

        self.__check = self.add_widget(CheckBox("Quad").set_value(True))

        self.start_horizontal()
        self.insert_text_label("Size: ")
        self.__size_h = self.add_widget(LineEdit())
        self.insert_text_label("X")
        self.__size_w = self.add_widget(LineEdit()).set_enabled(False)
        self.cancel()

        self.__check.add_clicked_callback(lambda: self.__size_w.set_enabled(not self.__check.get_value()))

    def _init_by_config(self, config: {}):
        size = config['size']
        if type(size) == int:
            self.__size_h.set_value(str(size))
        elif type(size) == list and len(size) == 2:
            self.__size_h.set_value(str(size[0]))
            self.__size_w.set_value(str(size[1]))
            self.__check.set_value(True)

    def get_config(self):
        return {'rcrop': {'percentage': 100,
                          'size': int(self.__size_h.get_value()) if self.__check.get_value() else
                          [int(self.__size_h.get_value()), int(self.__size_w.get_value())]}}


class RandomBrightnessUi(AbstractAugmentationUi):
    def __init__(self, dataset_path: str, project_dir_path: str, previous_augmentations: [] = None):
        super().__init__("Random Brightness", dataset_path, project_dir_path, previous_augmentations)

        self.start_horizontal()
        self.insert_text_label("Interval:")
        self.__from = self.add_widget(LineEdit())
        self.insert_text_label("X")
        self.__to = self.add_widget(LineEdit())
        self.cancel()

    def _init_by_config(self, config: {}):
        self.__from.set_value(str(config['interval'][0]))
        self.__to.set_value(str(config['interval'][1]))

    def get_config(self):
        return {'rbrightness': {'percentage': 100,
                                'interval': [int(self.__from.get_value()), int(self.__to.get_value())]}}


class RandomContrastUi(AbstractAugmentationUi):
    def __init__(self, dataset_path: str, project_dir_path: str, previous_augmentations: [] = None):
        super().__init__("Random Contrast", dataset_path, project_dir_path, previous_augmentations)

        self.start_horizontal()
        self.insert_text_label("Interval:")
        self.__from = self.add_widget(LineEdit())
        self.insert_text_label("X")
        self.__to = self.add_widget(LineEdit())
        self.cancel()

    def _init_by_config(self, config: {}):
        self.__from.set_value(str(config['interval'][0]))
        self.__to.set_value(str(config['interval'][1]))

    def get_config(self):
        return {'rcontrast': {'percentage': 100,
                              'interval': [int(self.__from.get_value()), int(self.__to.get_value())]}}


augmentations_ui = {"resize": ResizeUi,
                    "hflip": HorizontalFlipUi,
                    "vflip": VerticalFlipUi,
                    "gauss_noise": GaussNoiseUi,
                    "snp_noise": SNPNoiseUi,
                    "blur": BlurUi,
                    "rrotate": RandomRotateUi,
                    "ccrop": CentralCropUi,
                    "rcrop": RandomCropUi,
                    "rbrightness": RandomBrightnessUi,
                    "rcontrast": RandomContrastUi}
