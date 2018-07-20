import json
import os
from pathlib import Path

from PySide2.QtGui import QKeySequence
from PySide2.QtWidgets import QAction

from data_conveyor.augmentations import augmentations_dict
from data_processor.model import model_urls, start_modes
from .PySide2Wrapper.PySide2Wrapper import MainWindow, ComboBox, OpenFile, LineEdit, Button, SaveFile, CheckBox, \
    ListWidget, Widget, DynamicView, DockWidget, MessageWindow

from .augmentations import augmentations_ui
from utils.config import default_config


class ParamException(Exception):
    def __init__(self, message: str):
        super().__init__()
        self.__msg = message

    def __str__(self):
        return self.__msg


class DataProcessorUi:
    def __init__(self, parent: Widget):
        self.__optimizers = ['Adam', 'SGD']

        parent.start_group_box("Data Processor")
        self.__architecture = parent.add_widget(
            ComboBox().add_label('Architecture', 'top').add_items(model_urls.keys()))
        self.__optimizer = parent.add_widget(ComboBox().add_label('Optimizer', 'top').add_items(self.__optimizers))

        parent.start_group_box('Learning rate')
        self.__lr_start_value = parent.add_widget(LineEdit().add_label('Start value: ', 'left'))
        self.__lr_skip_steps_num = parent.add_widget(LineEdit().add_label('Skip steps number: ', 'left'))
        self.__lr_decrease_coefficient = parent.add_widget(LineEdit().add_label('Decrease coefficient: ', 'left'))
        self.__lr_first_epoch_decrease_coeff = parent.add_widget(
            LineEdit().add_label('First epoch decrease coeff: ', 'left'))
        parent.cancel()

        self.__start_from = parent.add_widget(ComboBox().add_label('Start from', 'top').add_items(start_modes))
        parent.cancel()

        self.__models = [k for k in model_urls.keys()]

    def init_by_config(self, config: {}):
        cur_config = config['data_processor']
        self.__architecture.set_value(self.__models.index(cur_config['architecture']))
        self.__optimizer.set_value(self.__optimizers.index(cur_config['optimizer']))
        self.__start_from.set_value(start_modes.index(cur_config['start_from']))

        self.__lr_start_value.set_value(str(cur_config['learning_rate']['start_value']))
        self.__lr_skip_steps_num.set_value(str(cur_config['learning_rate']['skip_steps_number']))
        self.__lr_decrease_coefficient.set_value(str(cur_config['learning_rate']['decrease_coefficient']))
        self.__lr_first_epoch_decrease_coeff.set_value(str(cur_config['learning_rate']['first_epoch_decrease_coeff']))

    def flush_to_config(self, config: {}):
        config['data_processor']['architecture'] = self.__models[int(self.__architecture.get_value())]
        config['data_processor']['optimizer'] = self.__optimizers[int(self.__optimizer.get_value())]
        config['data_processor']['start_from'] = start_modes[int(self.__start_from.get_value())]

        config['data_processor']['learning_rate'] = {}
        try:
            config['data_processor']['learning_rate']['start_value'] = float(self.__lr_start_value.get_value())
        except:
            raise ParamException('Please set learning rate start value')

        try:
            config['data_processor']['learning_rate']['skip_steps_number'] = int(self.__lr_skip_steps_num.get_value())
        except:
            raise ParamException('Please set learning rate skip steps number')

        try:
            config['data_processor']['learning_rate']['decrease_coefficient'] = float(
                self.__lr_decrease_coefficient.get_value())
        except:
            raise ParamException('Please set learning rate decrease coefficient')

        try:
            config['data_processor']['learning_rate']['first_epoch_decrease_coeff'] = float(
                self.__lr_first_epoch_decrease_coeff.get_value())
        except:
            raise ParamException('Please set learning rate first epoch decrease coefficient')


class AugmentationsUi(Widget):
    class AugmentationNotConfiguredException(Exception):
        def __init__(self, aug_name: str):
            self.__aug_name = aug_name

        def get_name(self):
            return self.__aug_name

    class AugmentationUi(Widget):
        def __init__(self, parent, aug_names: [], is_first=False):
            super().__init__()
            self.__dataset_path = None
            self.__project_dir_path = None

            self.start_horizontal()
            self.__combo = self.add_widget(ComboBox().add_items(aug_names), need_stretch=False)
            self.__aug_names = aug_names
            self.__augmentation_instance = None

            self.__add_btn = self.add_widget(Button('+', is_tool_button=True), need_stretch=False)
            self.__settings_btn = self.add_widget(
                Button('s', is_tool_button=True).set_on_click_callback(self.__configure).set_on_click_callback(
                    lambda: parent.augmentations_changed()),
                need_stretch=False)

            self.__del_btn = None
            if not is_first:
                self.__del_btn = self.add_widget(Button('-', is_tool_button=True), need_stretch=False)
                self.__del_btn.set_on_click_callback(self.delete)
                self.__del_btn.set_on_click_callback(lambda: parent.del_augmentation(self))

            self.cancel()
            self.__add_btn.set_on_click_callback(parent.add_augmentation)

            self.__previous_augmentations = None

        def set_dataset_path(self, datatset_path):
            self.__dataset_path = datatset_path

        def set_project_dir_path(self, project_dir_path):
            self.__project_dir_path = project_dir_path

        def delete(self):
            self._layout.deleteLater()
            if self.__del_btn is not None:
                self.__del_btn.get_instance().deleteLater()
            self.__settings_btn.get_instance().deleteLater()
            self.__add_btn.get_instance().deleteLater()
            self.__combo.get_instance().deleteLater()

        def set_previous_augmentations(self, previous_augmentations):
            self.__previous_augmentations = previous_augmentations

        def get_value(self):
            return self.__augmentation_instance

        def get_name(self):
            return self.__aug_names[self.__combo.get_value()]

        def set_value(self, val):
            self.__augmentation_instance = val
            self.__combo.set_value(self.__aug_names.index(val.get_name()))

        def __configure(self):
            ui = augmentations_ui[self.__aug_names[self.__combo.get_value()]](self.__dataset_path, self.__project_dir_path,
                                                                              self.__previous_augmentations)
            if self.__augmentation_instance is not None:
                ui.init_by_config(self.__augmentation_instance.get_config())
            self.__augmentation_instance = ui.show()

    def __init__(self):
        super().__init__()
        self.__dataset_path = None
        self.__project_dir_path = None

        self.__default_name = '- None -'
        self.__augs = []
        self.__augs_names = [self.__default_name] + [k for k in augmentations_dict.keys()]

        self.__augmentations_changed_callbacks = []
        self.add_augmentation(is_first=True)

    def set_dataset_path(self, dataset_path: str):
        self.__dataset_path = dataset_path

        for aug in self.__augs:
            aug.set_dataset_path(self.__dataset_path)

    def set_project_dir_path(self, project_dir_path):
        self.__project_dir_path = project_dir_path

        for aug in self.__augs:
            aug.set_project_dir_path(self.__project_dir_path)

    def set_previous_augmentations(self, previous_augmentations):
        for a in self.__augs:
            a.set_previous_augmentations(previous_augmentations)

    def add_augmenatations_changed_callback(self, callback: callable):
        self.__augmentations_changed_callbacks.append(callback)

    def augmentations_changed(self):
        for call in self.__augmentations_changed_callbacks:
            call()

    def add_augmentation(self, is_first=False):
        if (len(self.__augs) > 1 and self.__augs[-1].get_value() is None) or (
                len(self.__augs) == 1 and self.__augs[0].get_value() is None):
            return

        new_aug = self.AugmentationUi(self, self.__augs_names, is_first)
        self.__augs.append(new_aug)
        self.add_widget(new_aug)
        self.augmentations_changed()

    def del_augmentation(self, augmentation):
        del self.__augs[self.__augs.index(augmentation)]

    def get_augmentations(self):
        return [a.get_value() for a in self.__augs]

    def init_by_config(self, config: {}):
        if len(self.__augs) > 0:
            for a in self.__augs:
                a.delete()
            self.__augs = []

        for i, aug in enumerate(config):
            self.add_augmentation(i == 0)
            self.__augs[-1].set_value(augmentations_dict[next(iter(aug))](aug))

        self.augmentations_changed()

    def flush_to_config(self, config: {}):
        for aug in self.__augs:
            if aug.get_name() == self.__default_name:
                continue
            a = aug.get_value()
            if a is None:
                raise AugmentationsUi.AugmentationNotConfiguredException(aug.get_name())
            config.append(aug.get_value().get_config())


class DataConveyorStepUi(Widget):
    def __init__(self, step_name: str):
        super().__init__()
        self.__step_name = step_name.lower()

        self.__dataset_path = self.add_widget(
            OpenFile("Dataset file").set_files_types('*.json').set_value_changed_callback(self.__dataset_path_changed))
        self.__before_augs, self.__augs, self.__after_augs = self.__init_augs()
        self.start_horizontal()
        self.__aug_percentage = self.add_widget(LineEdit().add_label("Augmentations percentage", 'left'))
        self.__data_percentage = self.add_widget(LineEdit().add_label("Data percentage", 'left'))
        self.cancel()

    def __dataset_path_changed(self, path):
        self.__before_augs.set_dataset_path(path)
        self.__augs.set_dataset_path(path)
        self.__after_augs.set_dataset_path(path)

    def set_project_dir_path(self, project_dir_path):
        self.__before_augs.set_project_dir_path(project_dir_path)
        self.__augs.set_project_dir_path(project_dir_path)
        self.__after_augs.set_project_dir_path(project_dir_path)

    def get_dataset_path(self):
        return self.__dataset_path.get_value()

    def get_aug_percentage(self):
        return int(self.__aug_percentage.get_value())

    def get_data_percentage(self):
        return int(self.__data_percentage.get_value())

    def get_before_augs(self):
        return self.__before_augs

    def get_augs(self):
        return self.__augs

    def get_after_augs(self):
        return self.__after_augs

    def __init_augs(self):
        self.start_horizontal()

        self.start_group_box('Before augmentations')
        before_augs = self.add_widget(AugmentationsUi())
        self.cancel()

        self.start_group_box('Augmentations')
        augs = self.add_widget(AugmentationsUi())
        before_augs.add_augmenatations_changed_callback(lambda: augs.set_previous_augmentations(before_augs.get_augmentations()))
        self.cancel()

        self.start_group_box('After augmentations')
        after_augs = self.add_widget(AugmentationsUi())
        self.cancel()

        self.cancel()

        return before_augs, augs, after_augs

    def init_by_config(self, config: {}, project_path: str):
        cur_config = config['data_conveyor'][self.__step_name]
        self.__dataset_path.set_value(os.path.join(project_path, cur_config['dataset_path']))

        if 'augmentations_percentage' in cur_config:
            self.__aug_percentage.set_value(str(cur_config['augmentations_percentage']))
        self.__data_percentage.set_value(str(cur_config['images_percentage']))

        if 'before_augmentations' in cur_config:
            self.__before_augs.init_by_config(cur_config['before_augmentations'])
        if 'augmentations' in cur_config:
            self.__augs.init_by_config(cur_config['augmentations'])
            if 'before_augmentations' in cur_config:
                self.__augs.set_previous_augmentations(self.__before_augs.get_augmentations())
        if 'after_augmentations' in cur_config:
            self.__after_augs.init_by_config(cur_config['after_augmentations'])

        self.__dataset_path_changed(self.__dataset_path.get_value())
        self.set_project_dir_path(project_path)

    def flush_to_config(self, config: {}, project_path: str):
        config['data_conveyor'][self.__step_name] = {}
        config['data_conveyor'][self.__step_name]['dataset_path'] = os.path.relpath(self.__dataset_path.get_value(), project_path)
        config['data_conveyor'][self.__step_name]['augmentations_percentage'] = int(self.__aug_percentage.get_value())
        config['data_conveyor'][self.__step_name]['images_percentage'] = int(self.__data_percentage.get_value())

        config['data_conveyor'][self.__step_name]['before_augmentations'] = []
        config['data_conveyor'][self.__step_name]['augmentations'] = []
        config['data_conveyor'][self.__step_name]['after_augmentations'] = []

        class_name = None
        try:
            class_name = "Before augmentations"
            self.__before_augs.flush_to_config(config['data_conveyor'][self.__step_name]['before_augmentations'])
            class_name = "Augmentations"
            self.__augs.flush_to_config(config['data_conveyor'][self.__step_name]['augmentations'])
            class_name = "After augmentations"
            self.__after_augs.flush_to_config(config['data_conveyor'][self.__step_name]['after_augmentations'])
        except AugmentationsUi.AugmentationNotConfiguredException as err:
            msg = "Augmentation\n{}\nin\n{}.{}\nnot configured".format(err.get_name(), self.__step_name, class_name)
            MessageWindow("Neural Studio", msg).show()
            raise Exception(msg)


class DataConveyorUi:
    def __init__(self, parent: Widget):
        parent.start_group_box('Data Conveyor')
        parent.start_horizontal()
        parent.insert_text_label('Data size:')
        self.__dc_data_size_x = parent.add_widget(LineEdit())
        parent.insert_text_label('X')
        self.__dc_data_size_y = parent.add_widget(LineEdit())
        parent.insert_text_label('X')
        self.__dc_data_size_c = parent.add_widget(LineEdit())
        parent.cancel()
        parent.start_horizontal()
        self.__dc_batch_size = parent.add_widget(LineEdit().add_label('Batch size', 'top'))
        self.__dc_threads_num = parent.add_widget(LineEdit().add_label('Threads number', 'top'))
        self.__dc_epoch_num = parent.add_widget(LineEdit().add_label('Epochs number', 'top'))
        parent.cancel()

        parent.start_horizontal()
        self.__use_folds = parent.add_widget(CheckBox("Train by folds"))
        self.__folds_number = parent.add_widget(LineEdit().add_label("Folds number", 'left'))
        parent.cancel()
        self.__dataset_path = parent.add_widget(OpenFile("Dataset path"))
        parent.cancel()

        self.__folds_number.add_enabled_dependency(self.__use_folds)
        self.__dataset_path.add_enabled_dependency(self.__use_folds)

    def init_by_config(self, config: {}):
        cur_config = config['data_conveyor']
        self.__dc_data_size_x.set_value(str(cur_config['data_size'][0]))
        self.__dc_data_size_y.set_value(str(cur_config['data_size'][1]))
        self.__dc_data_size_c.set_value(str(cur_config['data_size'][2]))
        self.__dc_batch_size.set_value(str(cur_config['batch_size']))
        self.__dc_threads_num.set_value(str(cur_config['threads_num']))
        self.__dc_epoch_num.set_value(str(cur_config['epoch_num']))
        self.__use_folds.set_value(bool(cur_config['train_by_folds']))
        if self.__use_folds.get_value():
            self.__folds_number.set_value(str(cur_config['folds_number']))
            self.__dataset_path.set_value(str(cur_config['dataset_path']))

    def flush_to_config(self, config: {}):
        config['data_conveyor']['data_size'] = [int(self.__dc_data_size_x.get_value()),
                                                int(self.__dc_data_size_y.get_value()),
                                                int(self.__dc_data_size_c.get_value())]
        config['data_conveyor']['batch_size'] = int(self.__dc_batch_size.get_value())
        config['data_conveyor']['threads_num'] = int(self.__dc_threads_num.get_value())
        config['data_conveyor']['epoch_num'] = int(self.__dc_epoch_num.get_value())

        config['data_conveyor']['train_by_folds'] = bool(self.__use_folds.get_value())
        if self.__use_folds.get_value():
            config['data_conveyor']['folds_number'] = int(self.__folds_number.get_value())
            config['data_conveyor']['dataset_path'] = str(self.__dataset_path.get_value())


class NeuralStudio(MainWindow):
    class Config(Widget):
        def __init__(self, config, project_path):
            super().__init__()
            self.start_horizontal()
            self.__data_processor = DataProcessorUi(self)

            self.start_vertical()
            self.__data_conveyor = DataConveyorUi(self)

            self.insert_tab_space()
            self.add_tab("Train")
            self.__dc_train_step = self.add_widget(DataConveyorStepUi('Train'))
            self.cancel()
            self.add_tab("Validation")
            self.__dc_validation_step = self.add_widget(DataConveyorStepUi('Validation'))
            self.cancel()
            self.add_tab("Test")
            self.__dc_test_step = self.add_widget(DataConveyorStepUi('Test'))
            self.cancel()
            self.cancel()
            self.cancel()

            self.set_project_dir_path(project_path)
            self.init_by_config(config, project_path)

        def init_by_config(self, config: {}, project_path: str):
            self.__data_conveyor.init_by_config(config)
            self.__data_processor.init_by_config(config)
            self.__dc_train_step.init_by_config(config, project_path)
            self.__dc_validation_step.init_by_config(config, project_path)
            self.__dc_test_step.init_by_config(config, project_path)

            self.set_project_dir_path(project_path)

        def flush_to_config(self, config: {}, project_path: str):
            self.__data_conveyor.flush_to_config(config)
            self.__data_processor.flush_to_config(config)
            self.__dc_train_step.flush_to_config(config, project_path)
            self.__dc_validation_step.flush_to_config(config, project_path)
            self.__dc_test_step.flush_to_config(config, project_path)

        def set_project_dir_path(self, project_dir_path):
            self.__dc_train_step.set_project_dir_path(project_dir_path)
            self.__dc_validation_step.set_project_dir_path(project_dir_path)
            self.__dc_test_step.set_project_dir_path(project_dir_path)

    class ConfigsList(DockWidget):
        def __init__(self, parent):
            super().__init__("Configs", parent.get_instance())

            self.start_horizontal()
            add_btn = self.add_widget(Button("+", is_tool_button=True), need_stretch=False)
            del_btn = self.add_widget(Button("-", is_tool_button=True), need_stretch=False)
            self.cancel()
            self.__list_view = self.add_widget(ListWidget(), need_stretch=False)
            add_btn.set_on_click_callback(lambda: parent.add_config())
            del_btn.set_on_click_callback(lambda: parent.del_config(self.__list_view.get_current_idx()))
            self.__list_view.set_item_renamed_callback(lambda idx, new_name: parent.rename_config(idx, new_name))

        def get_instance(self):
            return self.__list_view

    def __init__(self):
        super().__init__("Neural Studio")

        self.__menu_bar = self.get_instance().menuBar()
        self.__file_menu = self.__menu_bar.addMenu('File')
        self.__open_project_act = QAction("Open project", self.__file_menu)
        self.__open_project_act.setShortcut(QKeySequence("Ctrl+O"))
        self.__open_project_act.triggered.connect(self.__open_project)
        self.__save_project_act = QAction("Save project", self.__file_menu)
        self.__save_project_act.setShortcut(QKeySequence("Ctrl+S"))
        self.__save_project_act.triggered.connect(self.__save_project)
        self.__clear_trash_act = QAction("Clear trash", self.__file_menu)
        self.__clear_trash_act.triggered.connect(self.__clear_trash)
        self.__clear_trash_act.setEnabled(False)
        self.__file_menu.addAction(self.__open_project_act)
        self.__file_menu.addAction(self.__save_project_act)
        self.__file_menu.addSeparator()
        self.__file_menu.addAction(self.__clear_trash_act)

        self.__last_config_id = 0

        self.__project_path = os.path.abspath(Path.home())
        self.__configs = [{"name": "config", "instance": self.Config(default_config, self.__project_path), 'id': 0}]

        self.start_horizontal()
        self.__chunks_items = self.add_widget(self.ConfigsList(self))
        self.__cur_view = self.add_widget(DynamicView())
        self.cancel()

        for c in self.__configs:
            self.__cur_view.add_item(c['instance'])

        self.__chunks_items.get_instance().add_items([c['name'] for c in self.__configs])
        self.__chunks_items.get_instance().set_value_changed_callback(self.__cur_config_changed)

        self.__project_config = None

    def del_config(self, idx):
        if idx is None:
            return
        self.__chunks_items.get_instance().remove_item(idx)
        self.__cur_view.remove_item(idx)
        del self.__configs[idx]

    def add_config(self, name='config', config=default_config, id=None):
        if id is None:
            self.__configs.append({"name": name, "instance": self.Config(config, self.__project_path), 'id': self.__last_config_id})
        else:
            cur_path = os.path.join(self.__project_path, 'workdir', str(id))
            self.__configs.append({"name": name, "instance": self.Config(config, cur_path), 'id': id})

            if id > self.__last_config_id:
                self.__last_config_id = id

        self.__chunks_items.get_instance().add_item(self.__configs[-1]['name'])
        self.__cur_view.add_item(self.__configs[-1]['instance'])

        self.__last_config_id += 1

    def rename_config(self, idx, new_name):
        if idx is not None:
            self.__configs[idx]['name'] = new_name

    def __cur_config_changed(self, idx):
        if idx is not None:
            self.__cur_view.set_index(idx)

    def __change_project_path(self, title: str, is_open=True):
        if is_open:
            res = OpenFile(title).set_files_types('*.ns').call()
        else:
            res = SaveFile(title).set_files_types('*.ns').call()

        if res is None or len(res) < 1:
            return False

        res = os.path.abspath(res)
        self.__project_path = os.path.dirname(res)
        self.__project_name = os.path.basename(res)

        return True

    def __project_path_changed(self):
        self.set_title_prefix(self.__project_name)

        for config in self.__configs:
            config['instance'].set_project_dir_path(self.__project_path)

    def __open_project(self):
        if not self.__change_project_path('Open project'):
            return

        with open(os.path.join(self.__project_path, self.__project_name), 'r') as file:
            self.__project_config = json.load(file)

        self.__close_project()

        for c in self.__project_config:
            config_id = c['id']
            config_path = os.path.join(self.__project_path, "workdir", str(config_id), 'config.json')
            with open(config_path, 'r') as file:
                config = json.load(file)
            self.add_config(name=c['name'], config=config, id=config_id)

        self.__project_path_changed()
        self.__clear_trash_act.setEnabled(True)

    def __save_project(self):
        if not self.__change_project_path('Save project', is_open=False):
            return

        project_config = [{"name": c['name'], "id": c['id']} for c in self.__configs]
        with open(os.path.join(self.__project_path, self.__project_name), 'w') as outfile:
            json.dump(project_config, outfile, indent=2)

        for c in self.__configs:
            cur_path = os.path.join(self.__project_path, 'workdir', str(c['id']))

            config = {'data_processor': {}, 'data_conveyor': {}}
            c['instance'].flush_to_config(config, cur_path)

            if not os.path.exists(cur_path) or not os.path.isdir(cur_path):
                os.makedirs(cur_path)
            with open(os.path.join(cur_path, 'config.json'), 'w') as outfile:
                json.dump(config, outfile, indent=2)

        self.__project_path_changed()

    def __close_project(self):
        if len(self.__configs) == 1:
            self.del_config(0)
        else:
            for i in range(len(self.__configs) - 1, 0, -1):
                self.del_config(i)
        self.__configs = []

    def __clear_trash(self):
        MessageWindow("Clear trash", "Not implemented", self.get_instance()).show()
