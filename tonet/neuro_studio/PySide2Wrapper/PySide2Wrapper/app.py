import sys

from PySide2.QtWidgets import *


class Application:
    """
    Class, that provide system information and interaction
    """
    def __init__(self):
        self.__app = QApplication(sys.argv)

    def run(self):
        """
        Run application
        :return:
        """
        return self.__app.exec_()

    def screen_resolution(self):
        """
        Get current screen size
        @return: screen
        """
        screen_geometry = self.__app.desktop().screenGeometry()
        return [screen_geometry.width(), screen_geometry.height()]

    def get_instance(self):
        return self.__app
