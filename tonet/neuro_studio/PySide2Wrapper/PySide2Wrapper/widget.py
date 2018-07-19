import os

from PySide2.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox, QRadioButton, \
    QComboBox, QProgressBar, QTableWidget, QHeaderView, QTableWidgetItem, QFileDialog, QToolButton, QTabWidget, \
    QWidget, QListWidget, QListWidgetItem, QGroupBox, QStackedLayout, QSplitter
from PySide2.QtGui import QPixmap, QImage, QDoubleValidator, QIntValidator, QRegExpValidator
from PySide2.QtCore import QObject, Signal, QDir, Qt
from abc import ABCMeta, abstractmethod

from .utils import StateSaver


class Checkable(metaclass=ABCMeta):
    @abstractmethod
    def add_clicked_callback(self, callback: callable):
        """
        Add callback, that will call for every click
        :param callback: callback
        :return: Widget object
        :rtype: Widget
        """

    @abstractmethod
    def set_value(self, state: bool):
        """
        Set state of Widget: checked or not
        :return: State
        :rtype: self instance
        """

    @abstractmethod
    def get_value(self):
        """
        Get state of Widget: checked or not
        :return: State
        :rtype: bool
        """


class Widget:
    def __init__(self, instance: QObject = None):
        self._layout = QVBoxLayout()
        if instance is not None:
            self._instance = instance
        self._enabled_dependencies = []
        self._state_saver = None
        self._layouts = [QVBoxLayout()]
        self._cur_tab_widget = None
        self._cur_splitter = None

    def set_state_saver(self, saver: StateSaver):
        self._state_saver = saver

    def get_layout(self):
        """
        Return layout of widget
        :return: layout, contains Widget instance
        @:rtype: QLayout
        """
        self._layout.addLayout(self._layouts[-1])
        return self._layout

    def get_instance(self):
        """
        Get Qt instance of widget
        :return: instance of widget (PySide object)
        :rtype QWidget
        """
        return self._instance

    def _may_be_enabled(self, is_enabled):
        if is_enabled:
            for depends in self._enabled_dependencies:
                if not depends.get_value():
                    return False
        return is_enabled

    def set_enabled(self, is_enabled: bool = True):
        """
        Set widget enabled
        :param is_enabled: state of widget: enabled or not
        :return: Widget object (self)
        :rtype: Widget
        """
        self._instance.setEnabled(self._may_be_enabled(is_enabled))
        return self

    def add_enabled_dependency(self, dependency: Checkable):
        self.set_enabled(dependency.get_value())
        self._enabled_dependencies.append(dependency)
        dependency.add_clicked_callback(self.set_enabled)

    def add_widget(self, widget: "Widget instance", need_store=False, need_stretch=True):
        """
        Add widget to window layout
        :param widget: Widget unit
        :param need_store: is need to store state  of specified widget
        :param need_stretch: is need to insert stretch around widget
        :return: widget instance
        """
        if need_stretch:
            self.get_current_layout().addStretch()
        self.get_current_layout().addLayout(widget.get_layout())
        if need_stretch:
            self.get_current_layout().addStretch()

        if need_store and self._state_saver is not None:
            self._state_saver.add_widget(widget)

        return widget

    def add_widgets(self, widgets: "list of Widget instances"):
        """
        Add list of widgets to window layout
        :param widgets: Widget units
        :return: None
        """
        for widget in widgets:
            self.get_current_layout().addLayout(widget.get_layout())

    def start_horizontal(self):
        """
        Start horizontal components insertion
        :return: None
        """
        if isinstance(self.get_current_layout(), QHBoxLayout):
            return
        layout = QHBoxLayout()
        self.get_current_layout().addLayout(layout)
        self._layouts.append(layout)

    def start_vertical(self):
        """
        Start vertical components insertion
        :return: None
        """
        if isinstance(self.get_current_layout(), QVBoxLayout):
            return
        layout = QVBoxLayout()
        self.get_current_layout().addLayout(layout)
        self._layouts.append(layout)

    def group_horizontal(self, widgets: list):
        """
        Place list of widgets horizontal
        :param widgets: list of widgets
        :return: None
        """
        self.start_horizontal()
        self.add_widgets(widgets)
        self.cancel()

    def group_vertical(self, widgets: "list of Widget instances"):
        """
        Place list of widgets vertical
        :param widgets: list of widgets
        :return: None
        """
        self.start_vertical()
        self.add_widgets(widgets)
        self.cancel()

    def start_splitter(self, orientation='vertical'):
        self._cur_splitter = QSplitter()
        self._cur_splitter.setOrientation(Qt.Orientation.Horizontal if orientation == 'horizontal' else Qt.Orientation.Vertical)
        self.get_current_layout().addWidget(self._cur_splitter)

    def add_splitter_space(self):
        self._layouts.append(QVBoxLayout())
        widget = QWidget()
        widget.setLayout(self.get_current_layout())
        self._cur_splitter.addWidget(widget)

    def get_current_layout(self):
        """
        Return current layout
        :return: layout of type QLayout
        """
        return self._layouts[-1]

    def add_to_group_box(self, group_name: str, widgets: "list of Widget instances"):
        """
        Place layout to group box
        :param group_name: name of group
        :param widgets: list of widgets, that been placed to group
        :return: None
        """
        self.start_group_box(group_name)
        for widget in widgets:
            self.add_widget(widget)
        self.cancel()

    def start_group_box(self, name: str):
        """
        Start group box
        :return:
        """
        group_box = QGroupBox(name)
        group_box_layout = QVBoxLayout()
        group_box.setLayout(group_box_layout)
        self.get_current_layout().addWidget(group_box)
        self._layouts.append(group_box_layout)

    def cancel(self):
        """
        Cnacel last format append
        :return: None
        """
        del self._layouts[-1]

    def insert_text_label(self, text, is_link=False):
        widget = QLabel(text)
        widget.setOpenExternalLinks(is_link)
        self.get_current_layout().addWidget(widget)

    def insert_tab_space(self):
        self._cur_tab_widget = QTabWidget()
        self.get_current_layout().addWidget(self._cur_tab_widget)

    def add_tab(self, name):
        self._layouts.append(QVBoxLayout())
        widget = QWidget()
        widget.setLayout(self.get_current_layout())
        self._cur_tab_widget.addTab(widget, name)


class LabeledWidget(Widget, metaclass=ABCMeta):
    def __init__(self, instance: QObject = None):
        super().__init__(instance)
        self.__is_assembled = False

    def add_label(self, text: str, position: str):
        """
        Add label to widget
        :param text: label text
        :param position: position of label ['top', 'left', 'right', 'bottom']
        :return: self value
        """
        if not (position in ['top', 'bottom', 'left', 'right']):
            raise Exception("Wrong position for label: " + text)

        if position in ['top', 'bottom']:
            self._layout = QVBoxLayout()
        elif position in ['left', 'right']:
            self._layout = QHBoxLayout()

        if position in ['top', 'left']:
            if position == 'top':
                self._layout.addStretch()
            self._layout.addWidget(QLabel(text))
            self._layout.addWidget(self._instance)
            if position == 'top':
                self._layout.addStretch()
        elif position in ['bottom', 'right']:
            if position == 'bottom':
                self._layout.addStretch()
            self._layout.addWidget(self._instance)
            self._layout.addWidget(QLabel(text))
            if position == 'bottom':
                self._layout.addStretch()

        self.__is_assembled = True
        return self

    def get_layout(self):
        """
        Return layout of widget
        :return: layout
        @:rtype: QLayout
        """
        if not self.__is_assembled:
            self._assembly()
        return self._layout

    @abstractmethod
    def _assembly(self):
        """
        Assembly object
        :return: object instance
        """


class ValueContains(metaclass=ABCMeta):
    @abstractmethod
    def set_value(self, value):
        """
        Set value to
        :param value: value to set
        :return: None
        """

    @abstractmethod
    def get_value(self):
        """
        Get value
        :return: current value
        """

    @abstractmethod
    def set_value_changed_callback(self, callback: callable):
        """
        Set callback? that called when value in widget changed
        :return: self object
        """


class LineEdit(LabeledWidget, ValueContains):
    def __init__(self):
        super().__init__(QLineEdit())

    def set_data_type(self, data_type: str, regexp: str = None):
        """
        Set data type? contains in LineEdit
        :param data_type: type of data. String with possible values: {'int', 'double', 'regexp'}. If set 'regexp' type, need to define regexp parameter
        :param regexp: regexp, that filter contain in LineEdit
        :return: self instance
        """
        if data_type == 'int':
            self._instance.setValidator(QIntValidator())
        elif data_type == 'double':
            self._instance.setValidator(QDoubleValidator())
        elif data_type == 'regexp':
            if regexp is None:
                raise Exception("Regexp doesn't specified for LineEdit")
            self._instance.setValidator(QRegExpValidator(regexp))
        else:
            raise Exception("Incorrect data_type parameter value: '{}' in LineEdit.set_data_type".format(data_type))

        return self

    def set_value(self, value: str):
        """
        Set value to
        :param value:
        :return:
        """
        self._instance.setText(value)

    def get_value(self):
        """
        Get current value
        :return: current value
        """
        return self._instance.text()

    def set_value_changed_callback(self, callback: callable):
        self._instance.textChanged.connect(callback)
        return self

    def _assembly(self):
        self._layout.addWidget(self._instance)


class Button(Widget):
    def __init__(self, title: str, is_tool_button: bool = False):
        super().__init__(QToolButton() if is_tool_button else QPushButton())
        self._layout.addWidget(self._instance)
        self._instance.setText(title)

    def set_on_click_callback(self, callback: callable):
        """
        Set callback on click event
        :param callback:
        :return:
        """
        self._instance.clicked.connect(callback)
        return self


class ImageLayout(Widget):
    def __init__(self):
        super().__init__(QLabel())
        self._layout.addWidget(self._instance)
        self.__pixmap = None

    def set_image_from_data(self, image, width, height, bytes_per_line):
        img = QImage(image, width, height, bytes_per_line, QImage.Format_RGB888)
        self.__pixmap = QPixmap.fromImage(img)
        self._instance.setPixmap(self.__pixmap)
        return self

    def set_image_from_file(self, file_path: str):
        self._instance.setPixmap(file_path)
        return self

    def set_size(self, width, height):
        self.__pixmap = self.__pixmap.scaledToWidth(width).scaledToHeight(height)
        self._instance.setPixmap(self.__pixmap)
        return self

    def get_size(self):
        return self.__pixmap.width(), self.__pixmap.height()


class CheckBox(Widget, Checkable):
    def __init__(self, title: str):
        super().__init__(QCheckBox(title))
        self._layout.addWidget(self._instance)

    def add_clicked_callback(self, callback: callable):
        self._instance.toggled.connect(callback)
        return self

    def set_value(self, state: bool):
        self._instance.setChecked(state)
        return self

    def get_value(self):
        return self._instance.isChecked()


class RadioButton(Widget, Checkable):
    def __init__(self, title: str):
        super().__init__(QRadioButton(title))
        self._layout.addWidget(self._instance)

    def set_value(self, state: bool):
        self._instance.setChecked(state)
        return self

    def get_value(self):
        return self._instance.isChecked()

    def add_clicked_callback(self, callback: callable):
        self._instance.toggled.connect(callback)
        return self


class ComboBox(LabeledWidget, ValueContains):
    def __init__(self):
        super().__init__(QComboBox())

    def _assembly(self):
        self._layout.addWidget(self._instance)

    def add_items(self, values: []):
        """
        Add values to combo box
        @param values: list of values
        @return: self instance
        """
        for v in values:
            self._instance.addItem(v)

        return self

    def set_value(self, value: int):
        self._instance.setCurrentIndex(value)

    def get_value(self):
        return self._instance.currentIndex()

    def set_value_changed_callback(self, callback: callable):
        self._instance.currentIndexChanged.connect(callback)
        return self


class ProgressBar(Widget, ValueContains):
    class Instance(QObject):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        value_changed = Signal(int, str)

        def set_value(self, value: int, status: str = ""):
            self.value_changed.emit(value, status)

    def __init__(self):
        super().__init__(QProgressBar())
        self._layout.addWidget(self._instance)

        self.__status = QLabel()
        self._layout.addWidget(self.__status)

        self.__InstanceCls = self.Instance()
        self.__InstanceCls.value_changed.connect(self.__set_value)

    def set_value(self, value: int, status: str = ""):
        self.__InstanceCls.value_changed.emit(value, status)

    def get_value(self):
        return self._instance.getValue()

    def set_value_changed_callback(self, callback: callable):
        self._instance.valueChanged.connect(callback)
        return self

    def __set_value(self, value: int, status: str = ""):
        self._instance.setValue(value)
        self.__status.setText(status)


class Table(Widget):
    def __init__(self):
        super().__init__(QTableWidget())
        self._instance.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._layout.addWidget(self._instance)

    def add_row(self, items: []):
        row_idx = self._instance.rowCount()
        self._instance.setRowCount(row_idx + 1)
        for i, item in enumerate(items):
            self._instance.setItem(row_idx, i, QTableWidgetItem(item))
        return self

    def del_row(self):
        current_row = self._instance.currentRow()
        self._instance.removeRow(current_row)
        return self

    def set_columns_headers(self, headers: []):
        self._instance.setColumnCount(len(headers))
        self._instance.setHorizontalHeaderLabels(headers)
        return self


class PathDialog(Widget, ValueContains, metaclass=ABCMeta):
    def __init__(self, label: str, button_label: str):
        super().__init__(QFileDialog())
        self.__label = label
        self.__button_label = button_label
        self._default_path = QDir.homePath()

        self.__value_changed_callbacks = []

        self.__line_edit = None
        self.__button = None

    def set_default_path(self, default_path: str):
        self._default_path = default_path
        return self

    def get_default_path(self):
        return self._default_path

    def get_value(self):
        res = self.__line_edit.get_value()
        return None if len(res) < 1 else res

    def set_value(self, value):
        if self.__line_edit is not None:
            self.__line_edit.set_value(value)
        self.set_default_path(os.path.dirname(value))

        for c in self.__value_changed_callbacks:
            c(value)

    def set_value_changed_callback(self, callback: callable):
        self.__value_changed_callbacks.append(callback)
        return self

    def _update_value(self) -> None:
        """
        Update path value by dialog
        """
        res = self.call()
        if res is None or len(res) < 1:
            return
        self.set_value(res)

    @abstractmethod
    def _call(self) -> str:
        """
        Call dialog window
        :return: result path
        """

    def call(self) -> str:
        path = self._call()
        if path is None or len(path) < 1:
            return None
        self.set_value(path)
        return path

    def get_layout(self):
        super().get_layout()
        self._layout.addWidget(QLabel(self.__label))

        self.__line_edit = LineEdit()
        h_layout = QHBoxLayout()
        h_layout.addLayout(self.__line_edit.get_layout())
        self.__button = Button(self.__button_label, is_tool_button=True).set_on_click_callback(self._update_value)
        h_layout.addLayout(self.__button.get_layout())
        self._layout.addLayout(h_layout)
        return self._layout

    def set_enabled(self, is_enabled: bool = True):
        res = self._may_be_enabled(is_enabled)
        if self.__line_edit is not None:
            self.__line_edit.set_enabled(res)
        if self.__button is not None:
            self.__button.set_enabled(res)


class OpenFile(PathDialog):
    def __init__(self, label: str):
        super().__init__(label, "...")
        self.__files_types = ""

    def set_files_types(self, types: str):
        self.__files_types = types
        return self

    def _call(self):
        return self._instance.getOpenFileName(caption="Open File", dir=self._default_path, filter=self.__files_types)[0]


class SaveFile(PathDialog):
    def __init__(self, label: str):
        super().__init__(label, "...")
        self.__files_types = ""

    def set_files_types(self, types: str):
        self.__files_types = types
        return self

    def _call(self):
        return self._instance.getSaveFileName(caption="Save File", dir=self._default_path, filter=self.__files_types)[0]


class OpenDirectory(PathDialog):
    def __init__(self, label: str):
        super().__init__(label, "...")

    def _call(self):
        return self._instance.getExistingDirectory(caption="Open Directory", dir=self._default_path,
                                                   options=QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)


class ListWidget(Widget, ValueContains):
    def __init__(self):
        super().__init__(QListWidget())
        self._layout.addWidget(self._instance)
        self.__items = []

    def add_item(self, item: str, is_editable=True):
        self.__items.append(QListWidgetItem(item, self._instance))
        if is_editable:
            self.__items[-1].setFlags(self.__items[-1].flags() | Qt.ItemIsEditable)
        self._instance.addItem(self.__items[-1])

    def add_items(self, items: [str], is_editable=True) -> Widget:
        for item in items:
            self.add_item(item, is_editable)
        return self

    def remove_item(self, idx: int):
        self._instance.takeItem(idx)
        del self.__items[idx]

    def remove_current(self):
        self.remove_item(self.get_current_idx())

    def get_current_idx(self):
        return self.__get_item_idx(self._instance.currentItem())

    def clear(self):
        for i in range(len(self.__items)):
            self.remove_item(i)

    def set_value(self, value: int):
        self._instance.setItem(self.__items[value])

    def get_value(self):
        return self.__items.index(self._instance.currentItem())

    def set_value_changed_callback(self, callback: callable):
        self._instance.currentItemChanged.connect(lambda cur, prev: callback(self.__get_item_idx(cur)))
        return self

    def set_item_renamed_callback(self, callback: callable):
        def internal(item: QListWidgetItem):
            idx = self.__get_item_idx(self._instance.currentItem())
            callback(idx, item.text())

        self._instance.itemChanged.connect(internal)

    def __get_item_idx(self, item: QListWidgetItem):
        for idx, it in enumerate(self.__items):
            if it is item:
                return idx
        return None


class DynamicView(Widget):
    def __init__(self):
        super().__init__()
        self.__stacked_layout = QStackedLayout()
        self._layout.addLayout(self.__stacked_layout)

        self.__widgets = []

    def add_item(self, item: Widget):
        w = QWidget()
        w.setLayout(item.get_layout())
        self.__stacked_layout.addWidget(w)
        self.__widgets.append(w)

    def remove_item(self, idx: int):
        self.__stacked_layout.takeAt(idx)
        del self.__widgets[idx]

    def clear(self):
        for i in range(len(self.__widgets) - 1):
            self.remove_item(i)

    def set_index(self, idx: int):
        self.__stacked_layout.setCurrentIndex(idx)
