from abc import ABCMeta, abstractmethod

from PySide2.QtCore import Qt
from PySide2.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QDialog, QWidget, QLabel, QDockWidget, \
    QScrollArea, QMainWindow, QTabWidget

from .widget import Widget, Button, ProgressBar


class AbstractWindow(Widget, metaclass=ABCMeta):
    def __init__(self, title: str, instance: QWidget, enable_scrolling: bool = False):
        super().__init__(instance)

        self._instance.closeEvent = lambda event: self.__on_close(event)

        if enable_scrolling:
            self.__scroll = QScrollArea(self._instance)

            self._viewport = QWidget(self.__scroll)
            self.__layout = QVBoxLayout(self._viewport)
            self.__layout.setMargin(0)
            self.__layout.setSpacing(0)
            self._viewport.setLayout(self.__layout)

            self.__scroll.setWidget(self._viewport)
            self.__scroll.setWidgetResizable(True)

            self._instance.setWidget(self.__scroll)

            self._layouts = [self.__layout]
            self._window_layout = QVBoxLayout(self._instance)
            self._window_layout.setMargin(0)
            self._window_layout.setSpacing(0)
            self._instance.setLayout(self._window_layout)
        else:
            if not type(self._instance) in [QMainWindow, QWidget, QDialog]:
                widget = QWidget()
                self._layouts = [QVBoxLayout()]
                widget.setLayout(self.get_current_layout())
                self._instance.setWidget(widget)
            else:
                self._layouts = [QVBoxLayout()]
                self._instance.setLayout(self.get_current_layout())

        self.__title = title
        self._instance.setWindowTitle(self.__title)
        self._instance.resize(0, 0)

        self.__on_close_callbacks = []

    @abstractmethod
    def _show(self):
        """
        Internal method, that called from show()
        :return:
        """

    def show(self):
        """
        Show this window
        :return:
        """
        if self._state_saver is not None:
            self._state_saver.load()

        self._show()

    def add_on_close_callback(self, callback: callable):
        self.__on_close_callbacks.append(callback)

    def __on_close(self, event):
        for c in self.__on_close_callbacks:
            c(event)

        if self._state_saver is not None:
            self._state_saver.write()

    def set_title_prefix(self, prefix: str):
        """
        Set window title prefix. If empty string: reset title to window title
        :param prefix: window name prefix
        :return:
        """
        self._instance.setWindowTitle("[{}] - {}".format(prefix, self.__title) if prefix != "" else self.__title)

    def add_subwindow(self, title: str, is_modal=True):
        """
        Create subwindow
        :param title: window title
        :return: window
        @rtype Window
        """
        return ModalWindow(title, self._instance) if is_modal else Window(title, self._instance)

    def resize(self, width, height):
        self._instance.resize(width, height)

    def move(self, x, y):
        self._instance.move(x, y)

    def close(self):
        self._instance.close()


class Window(AbstractWindow):
    def __init__(self, title: str = "", parent=None):
        super().__init__(title, QWidget(parent))
        self._instance.setAttribute(Qt.WA_QuitOnClose, False)
        self._instance.setWindowFlags(self._instance.windowFlags() & (~Qt.WindowContextHelpButtonHint))

    def _show(self):
        self._instance.show()


class ModalWindow(AbstractWindow):
    def __init__(self, title: str = "", parent=None):
        super().__init__(title, QDialog(parent))
        self._instance.setWindowFlags(self._instance.windowFlags() & (~Qt.WindowContextHelpButtonHint))

    def _show(self):
        self._instance.exec_()


class MainWindow(AbstractWindow):
    def __init__(self, title: str = ""):
        super().__init__(title, QMainWindow())

        self._instance.setGeometry(0, 0, 0, 0)
        central_widget = QWidget()
        central_widget.setLayout(self.get_current_layout())
        self._instance.setCentralWidget(central_widget)

    def _show(self):
        self._instance.show()


class DockWidget(AbstractWindow):
    def __init__(self, title: str, parent, area: str = 'left'):
        """
        DockWidget initial constructor
        :param title: title of DockWidget
        :param parent: parent window
        :param area: area of placing; may be ['left', 'right', 'bottom', 'top']
        """
        super().__init__(title, QDockWidget(parent), False)

        areas = {'left': Qt.LeftDockWidgetArea, 'right': Qt.RightDockWidgetArea, 'bottom': Qt.BottomDockWidgetArea,
                 'top': Qt.TopDockWidgetArea}

        parent.addDockWidget(areas[area], self._instance)
        self.__parent = parent
        self._instance.show()

    def tabify(self, dock: QDockWidget):
        """
        Align dock widget with existing
        :param dock: dock widget
        """
        self.__parent.tabifyDockWidget(dock, self._instance)

    def _show(self):
        self._instance.show()


class MessageWindow(ModalWindow):
    def __init__(self, title: str, message: str = None, parent=None):
        super().__init__(title, parent=parent)

        if message is not None:
            self.insert_text_label(message)

        self.__btn = self.add_widget(Button("Ok").set_on_click_callback(lambda: self.close()))


class DialogWindow(ModalWindow):
    def __init__(self, title: str, buttons: [str], message: str = None, parent=None):
        super().__init__(title, parent=parent)

        if message is not None:
            self.insert_text_label(message)

        self.__layout = QVBoxLayout()
        self.__inner_layouts = [self.__layout]
        self.get_current_layout().addLayout(self.__layout)

        self.__choose = None
        self.start_horizontal()
        for button in buttons:
            self.__add_method(button)
            callback = getattr(self, button)
            setattr(self, button, self.add_widget(Button(button)).set_on_click_callback(lambda: self.close()).set_on_click_callback(callback))
        self.cancel()

        self.get_current_layout = lambda: self.__inner_layouts[-1]

    def show(self):
        super().show()
        return self.__choose

    def __add_method(self, rvalue):
        def innerdynamo():
            self.__choose = rvalue

        innerdynamo.__name__ = rvalue
        setattr(self, innerdynamo.__name__, innerdynamo)


class ProgressWindow(ModalWindow):
    def __init__(self, title: str):
        super().__init__(title)

        self.__progress_bar = self.add_widget(ProgressBar())
        self.__btn = self.add_widget(Button("Cancel")).set_on_click_callback(lambda: self.close())

    def show(self):
        super().show()

    def set_value(self, value: int, status: str = ""):
        self.__progress_bar.set_value(value, status)


class DoubleProgressWindow(ModalWindow):
    def __init__(self, title: str):
        super().__init__(title)

        self.__overall_pbar = self.add_widget(ProgressBar())
        self.__pbar = self.add_widget(ProgressBar())
        self.__btn = self.add_widget(Button("Cancel")).set_on_click_callback(lambda: self.close())

    def show(self):
        super().show()

    def set_overall_value(self, value: int, status: str = ""):
        self.__pbar.set_value(value, status)

    def set_value(self, value: int, status: str = ""):
        self.__overall_pbar.set_value(value, status)
