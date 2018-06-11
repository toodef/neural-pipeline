import cv2

from data_conveyor.augmentations import augmentations_dict
from neuro_studio.PySide2Wrapper.PySide2Wrapper import ImageLayout, CheckBox, LineEdit, ModalWindow, ABCMeta, \
    abstractmethod, Button


class AbstractAugmentationUi(ModalWindow, metaclass=ABCMeta):
    def __init__(self, aug_name):
        super().__init__(aug_name)

        self._path = r"C:\workspace\projects\nn\furniture_segmentation\workdir\train\17\180869.jpg"
        self._image = cv2.imread(self._path)

        self.add_widget(Button("Update").set_on_click_callback(self.update))
        self._percentage = self.add_widget(LineEdit().add_label("Percentage", 'left'))
        self._image_layout = self.add_widget(ImageLayout().set_image_from_file(self._path))

    def update(self):
        try:
            config = self.get_config()
            name = [k for k, v in config.items()][0]
            img = augmentations_dict[name](self.get_config())(self._image)
        except ValueError:
            return
        self._image_layout.set_image_from_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), img.shape[1], img.shape[0],
                                               img.shape[1] * 3)

    @abstractmethod
    def get_config(self):
        pass


class ResizeUi(AbstractAugmentationUi):
    def __init__(self):
        super().__init__("Resize")

        self.__check = self.add_widget(CheckBox("By minimum edge").set_value(True))

        self.start_horizontal()
        self.insert_text_label("Size: ")
        self.__size_h = self.add_widget(LineEdit())
        self.insert_text_label("X")
        self.__size_w = self.add_widget(LineEdit()).set_enabled(False)
        self.__check.add_clicked_callback(lambda: self.__size_w.set_enabled(not self.__check.get_value()))
        self.cancel()

    @abstractmethod
    def get_config(self):
        return {
            'resize': [int(self.__size_h.get_value()), int(self.__size_w.get_value())] if self.__check.get_value() else
            int(self.__size_h.get_value())}


class HorizontalFlipUi(AbstractAugmentationUi):
    def __init__(self):
        super().__init__("Horizontal Flip")

    def get_config(self):
        return {'hflip': {'precentage': 100}}


class VerticalFlipUi(AbstractAugmentationUi):
    def __init__(self):
        super().__init__("Horizontal Flip")

    def get_config(self):
        return {'vflip': {'precentage': 100}}


class GaussNoiseUi(AbstractAugmentationUi):
    def __init__(self):
        super().__init__("Gauss Noise")

        self.start_horizontal()
        self.__mean = self.add_widget(LineEdit().add_label("Mean", 'left').set_value_changed_callback(self.update))
        self.__var = self.add_widget(LineEdit().add_label("Var", 'left').set_value_changed_callback(self.update))
        self.__interval = self.add_widget(
            LineEdit().add_label("Interval", 'left').set_value_changed_callback(self.update))
        self.cancel()

    def get_config(self):
        return {'gauss_noise': {'percentage': 100,
                                'mean': float(self.__mean.get_value()),
                                'var': float(self.__var.get_value()),
                                'interval': int(self.__interval.get_value())}}


class SNPNoiseUi(AbstractAugmentationUi):
    def __init__(self):
        super().__init__("Salt&Paper Noise")

        self.start_horizontal()
        self.__s_vs_p = self.add_widget(
            LineEdit().add_label("Salt vs paper", 'left').set_value_changed_callback(self.update))
        self.__amount = self.add_widget(LineEdit().add_label("Amount", 'left').set_value_changed_callback(self.update))
        self.cancel()

    def get_config(self):
        return {'snp_noise': {'percentage': 100,
                              's_vs_p': float(self.__s_vs_p.get_value()),
                              'amount': float(self.__amount.get_value())}}


class BlurUi(AbstractAugmentationUi):
    def __init__(self):
        super().__init__("Blur")

        self.start_horizontal()
        self.insert_text_label("Kernel size: ")
        self.__ksize_x = self.add_widget(LineEdit().set_value_changed_callback(self.update))
        self.insert_text_label("X")
        self.__ksize_y = self.add_widget(LineEdit().set_value_changed_callback(self.update))
        self.cancel()

    def get_config(self):
        return {'blur': {'percentage': 100,
                         'ksize': (int(self.__ksize_x.get_value()),
                                   int(self.__ksize_y.get_value()))}}


class RandomRotateUi(AbstractAugmentationUi):
    def __init__(self):
        super().__init__("Random Rotate")

        self.start_horizontal()
        self.insert_text_label("Interval, [deg]: ")
        self.__interval_from = self.add_widget(LineEdit().set_value_changed_callback(self.update))
        self.insert_text_label("X")
        self.__interval_to = self.add_widget(LineEdit().set_value_changed_callback(self.update))
        self.cancel()

    def get_config(self):
        return {'rrotate': {'percentage': 100,
                            'interval': [int(self.__interval_from.get_value()), int(self.__interval_to.get_value())]}}


class CentralCropUi(AbstractAugmentationUi):
    def __init__(self):
        super().__init__("Central Crop")

        self.__check = self.add_widget(CheckBox("Quad").set_value(True))

        self.start_horizontal()
        self.insert_text_label("Size: ")
        self.__size_h = self.add_widget(LineEdit())
        self.insert_text_label("X")
        self.__size_w = self.add_widget(LineEdit()).set_enabled(False)
        self.cancel()

        self.__check.add_clicked_callback(lambda: self.__size_w.set_enabled(not self.__check.get_value()))

    def get_config(self):
        return {'ccrop': {'percentage': 100,
                          'size': int(self.__size_h.get_value()) if self.__check.get_value() else
                          [int(self.__size_h.get_value()), int(self.__size_w.get_value())]}}


class RandomCropUi(AbstractAugmentationUi):
    def __init__(self):
        super().__init__("Random Crop")

        self.__check = self.add_widget(CheckBox("Quad").set_value(True))

        self.start_horizontal()
        self.insert_text_label("Size: ")
        self.__size_h = self.add_widget(LineEdit())
        self.insert_text_label("X")
        self.__size_w = self.add_widget(LineEdit()).set_enabled(False)
        self.cancel()

        self.__check.add_clicked_callback(lambda: self.__size_w.set_enabled(not self.__check.get_value()))

    def get_config(self):
        return {'rcrop': {'percentage': 100,
                          'size': int(self.__size_h.get_value()) if self.__check.get_value() else
                          [int(self.__size_h.get_value()), int(self.__size_w.get_value())]}}


class RandomBrightnessUi(AbstractAugmentationUi):
    def __init__(self):
        super().__init__("Random Brightness")

        self.start_horizontal()
        self.insert_text_label("Interval:")
        self.__from = self.add_widget(LineEdit())
        self.insert_text_label("X")
        self.__to = self.add_widget(LineEdit())
        self.cancel()

    def get_config(self):
        return {'rbrightness': {'percentage': 100,
                                'interval': [int(self.__from.get_value()), int(self.__to.get_value())]}}


class RandomContrastUi(AbstractAugmentationUi):
    def __init__(self):
        super().__init__("Random Contrast")

        self.start_horizontal()
        self.insert_text_label("Interval:")
        self.__from = self.add_widget(LineEdit())
        self.insert_text_label("X")
        self.__to = self.add_widget(LineEdit())
        self.cancel()

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
