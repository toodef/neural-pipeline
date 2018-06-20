import json
import os

from PySide2.QtWidgets import QAction, QVBoxLayout, QHBoxLayout

from data_conveyor.augmentations import augmentations_dict
from data_processor.model import model_urls, start_modes
from neuro_studio.PySide2Wrapper.PySide2Wrapper import MainWindow, ComboBox, OpenFile, LineEdit, Button, SaveFile, \
    TabWidget, CheckBox, ListWidget

from neuro_studio.PySide2Wrapper.PySide2Wrapper import Application
from neuro_studio.augmentations import augmentations_ui
from utils.config import default_config


class ParamException(Exception):
    def __init__(self, message: str):
        super().__init__()
        self.__msg = message

    def __str__(self):
        return self.__msg


class DataProcessorUi:
    def __init__(self, window: MainWindow):
        self.__optimizers = ['Adam', 'SGD']

        window.start_group_box("Data Processor")
        self.__architecture = window.add_widget(
            ComboBox().add_label('Architecture', 'top').add_items(model_urls.keys()))
        self.__optimizer = window.add_widget(ComboBox().add_label('Optimizer', 'top').add_items(self.__optimizers))

        window.start_group_box('Learning rate')
        self.__lr_start_value = window.add_widget(LineEdit().add_label('Start value: ', 'left'))
        self.__lr_skip_steps_num = window.add_widget(LineEdit().add_label('Skip steps number: ', 'left'))
        self.__lr_decrease_coefficient = window.add_widget(LineEdit().add_label('Decrease coefficient: ', 'left'))
        self.__lr_first_epoch_decrease_coeff = window.add_widget(
            LineEdit().add_label('First epoch decrease coeff: ', 'left'))
        window.cancel()

        self.__start_from = window.add_widget(ComboBox().add_label('Start from', 'top').add_items(start_modes))
        window.cancel()

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


class AugmentationsUi:
    class AugmentationUi:
        def __init__(self, parent, aug_names: [], is_first=False):
            self.__combo = ComboBox().add_items(aug_names)
            self.__aug_names = aug_names
            self.__augmentation_instance = None

            self.__add_btn = Button('+', is_tool_button=True)
            self.__settings_btn = Button('s', is_tool_button=True).set_on_click_callback(self.__configure) \
                .set_on_click_callback(lambda: parent.augmentations_changed())
            self.__layout = QHBoxLayout()

            self.__del_btn = None
            if not is_first:
                self.__del_btn = Button('-', is_tool_button=True)
                self.__layout.addLayout(self.__del_btn.get_layout())
                self.__del_btn.set_on_click_callback(self.delete)
                self.__del_btn.set_on_click_callback(lambda: parent.del_augmentation(self))

            self.__layout.addLayout(self.__combo.get_layout())
            self.__layout.addLayout(self.__add_btn.get_layout())
            self.__layout.addLayout(self.__settings_btn.get_layout())
            parent.get_layout().addLayout(self.__layout)
            self.__add_btn.set_on_click_callback(parent.add_augmentation)

            self.__previous_augmentations = []

        def delete(self):
            self.__layout.deleteLater()
            if self.__del_btn is not None:
                self.__del_btn.get_instance().deleteLater()
            self.__settings_btn.get_instance().deleteLater()
            self.__add_btn.get_instance().deleteLater()
            self.__combo.get_instance().deleteLater()

        def set_previous_augmentations(self, previous_augmentations):
            self.__previous_augmentations = previous_augmentations

        def get_value(self):
            return self.__augmentation_instance

        def set_value(self, val):
            self.__augmentation_instance = val
            self.__combo.set_value(self.__aug_names.index(val.get_name()))

        def __configure(self):
            ui = augmentations_ui[self.__aug_names[self.__combo.get_value()]](self.__previous_augmentations)
            if self.__augmentation_instance is not None:
                ui.init_by_config(self.__augmentation_instance.get_config())
            self.__augmentation_instance = ui.show()

    def __init__(self, layout):
        self.__layout = layout
        self.__augs = []
        self.__augs_names = ['- None -'] + [k for k in augmentations_dict.keys()]

        self.__augmentations_changed_callbacks = []
        self.add_augmentation(is_first=True)

    def set_previous_augmentations(self, previous_augmentations):
        for a in self.__augs:
            a.set_previous_augmentations(previous_augmentations)

    def add_augmenatations_changed_callback(self, callback: callable):
        self.__augmentations_changed_callbacks.append(callback)

    def get_layout(self):
        return self.__layout

    def augmentations_changed(self):
        for call in self.__augmentations_changed_callbacks:
            call()

    def add_augmentation(self, is_first=False):
        if not is_first and self.__augs[-1].get_value() is None:
            return

        self.__augs.append(self.AugmentationUi(self, self.__augs_names, is_first))
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

        i = 0
        for aug in config:
            self.add_augmentation(i == 0)
            self.__augs[-1].set_value(augmentations_dict[next(iter(aug))](aug))
            i += 1

        self.augmentations_changed()

    def flush_to_config(self, config: {}):
        for aug in self.__augs:
            a = aug.get_value()
            if a is not None:  # TODO: sometimes None is happens, why?
                config.append(aug.get_value().get_config())


class DataConveyorStepUi:
    def __init__(self, step_name: str, window: MainWindow):
        self.__window = window
        self.__step_name = step_name.lower()

        self.__dataset_path = window.add_widget(OpenFile("Dataset file").set_files_types('*.json'))
        self.__before_augs, self.__augs, self.__after_augs = self.__init_augs()
        window.start_horizontal()
        self.__aug_percentage = window.add_widget(LineEdit().add_label("Augmentations percentage", 'left'))
        self.__data_percentage = window.add_widget(LineEdit().add_label("Data percentage", 'left'))
        window.cancel()

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
        self.__window.start_horizontal()
        self.__window.start_group_box('Before augmentations')
        layout = QVBoxLayout()
        self.__window.get_current_layout().addLayout(layout)
        before_augs = AugmentationsUi(layout)
        self.__window.cancel()
        self.__window.start_group_box('Augmentations')
        layout = QVBoxLayout()
        self.__window.get_current_layout().addLayout(layout)
        augs = AugmentationsUi(layout)
        before_augs.add_augmenatations_changed_callback(lambda: augs.set_previous_augmentations(before_augs.get_augmentations()))
        self.__window.cancel()
        self.__window.start_group_box('After augmentations')
        layout = QVBoxLayout()
        self.__window.get_current_layout().addLayout(layout)
        after_augs = AugmentationsUi(layout)
        self.__window.cancel()
        self.__window.cancel()

        return before_augs, augs, after_augs

    def init_by_config(self, config):
        cur_config = config['data_conveyor'][self.__step_name]
        self.__dataset_path.set_value(cur_config['dataset_path'])
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

    def flush_to_config(self, config: {}):
        config['data_conveyor'][self.__step_name] = {}
        config['data_conveyor'][self.__step_name]['dataset_path'] = self.__dataset_path.get_value()
        config['data_conveyor'][self.__step_name]['augmentations_percentage'] = int(self.__aug_percentage.get_value())
        config['data_conveyor'][self.__step_name]['images_percentage'] = int(self.__data_percentage.get_value())

        config['data_conveyor'][self.__step_name]['before_augmentations'] = []
        config['data_conveyor'][self.__step_name]['augmentations'] = []
        config['data_conveyor'][self.__step_name]['after_augmentations'] = []
        self.__before_augs.flush_to_config(config['data_conveyor'][self.__step_name]['before_augmentations'])
        self.__augs.flush_to_config(config['data_conveyor'][self.__step_name]['augmentations'])
        self.__after_augs.flush_to_config(config['data_conveyor'][self.__step_name]['after_augmentations'])


class DataConveyorUi:
    def __init__(self, window: MainWindow):
        window.start_group_box('Data Conveyor')
        window.start_horizontal()
        window.insert_text_label('Data size:')
        self.__dc_data_size_x = window.add_widget(LineEdit())
        window.insert_text_label('X')
        self.__dc_data_size_y = window.add_widget(LineEdit())
        window.insert_text_label('X')
        self.__dc_data_size_c = window.add_widget(LineEdit())
        window.cancel()
        window.start_horizontal()
        self.__dc_batch_size = window.add_widget(LineEdit().add_label('Batch size', 'top'))
        self.__dc_threads_num = window.add_widget(LineEdit().add_label('Threads number', 'top'))
        self.__dc_epoch_num = window.add_widget(LineEdit().add_label('Epochs number', 'top'))
        window.cancel()

        window.start_horizontal()
        self.__use_folds = window.add_widget(CheckBox("Train by folds"))
        self.__folds_number = window.add_widget(LineEdit().add_label("Folds number", 'left'))
        window.cancel()
        self.__dataset_path = window.add_widget(OpenFile("Dataset path"))
        window.cancel()

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
    def __init__(self):
        super().__init__("Neural Studio")

        self.__menu_bar = self.get_instance().menuBar()
        self.__file_menu = self.__menu_bar.addMenu('File')
        self.__open_project_act = QAction("Open project", self.__file_menu)
        self.__open_project_act.triggered.connect(self.__open_project)
        self.__save_project_act = QAction("Save project", self.__file_menu)
        self.__save_project_act.triggered.connect(self.__save_project)
        self.__file_menu.addAction(self.__open_project_act)
        self.__file_menu.addAction(self.__save_project_act)

        config = default_config

        self.start_horizontal()

        self.__chunks = self.add_widget(ListWidget())
        self.__chunks.add_items(["config"])

        self.__data_processor = DataProcessorUi(self)
        self.__data_processor.init_by_config(config)

        self.start_vertical()
        self.__data_conveyor = DataConveyorUi(self)
        self.__data_conveyor.init_by_config(config)

        self.insert_tab_space()
        self.add_tab("Train")
        self.__dc_train_step = DataConveyorStepUi('Train', self)
        self.__dc_train_step.init_by_config(config)
        self.cancel()
        self.add_tab("Validation")
        self.__dc_validation_step = DataConveyorStepUi('Validation', self)
        self.__dc_validation_step.init_by_config(config)
        self.cancel()
        self.add_tab("Test")
        self.__dc_test_step = DataConveyorStepUi('Test', self)
        self.__dc_test_step.init_by_config(config)
        self.cancel()
        self.cancel()

        self.cancel()

    def __change_project_path(self, title: str, is_open=True):
        if is_open:
            res = OpenFile(title).set_files_types('*.ns').call()
        else:
            res = SaveFile(title).set_files_types('*.ns').call()

        if len(res) < 1:
            return False

        self.__project_path = os.path.abspath(res)
        self.__project_name = os.path.basename(self.__project_path)

        return True

    def __open_project(self):
        if not self.__change_project_path('Open project'):
            return

        with open(self.__project_path, 'r') as file:
            config = json.load(file)

        self.__data_conveyor.init_by_config(config)
        self.__data_processor.init_by_config(config)
        self.__dc_train_step.init_by_config(config)
        self.__dc_validation_step.init_by_config(config)
        self.__dc_test_step.init_by_config(config)

        self.set_title_prefix(self.__project_name)

    def __save_project(self):
        if not self.__change_project_path('Save project', is_open=False):
            return

        config = {'data_processor': {}, 'data_conveyor': {}}
        self.__data_conveyor.flush_to_config(config)
        self.__data_processor.flush_to_config(config)
        self.__dc_train_step.flush_to_config(config)
        self.__dc_validation_step.flush_to_config(config)
        self.__dc_test_step.flush_to_config(config)

        with open(self.__project_path, 'w') as outfile:
            json.dump(config, outfile, indent=2)

        self.set_title_prefix(self.__project_name)


if __name__ == "__main__":
    app = Application()
    studio = NeuralStudio()
    resolution = app.screen_resolution()
    studio.resize(0, 0)
    studio.move(100, 100)
    studio.show()
    app.run()
