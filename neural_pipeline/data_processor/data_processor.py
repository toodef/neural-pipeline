import json
import numpy as np

import torch
from tqdm import tqdm

from ..data_processor.model import Model
from ..data_processor.state_manager import StateManager
from ..train_pipeline.train_pipeline import TrainPipeline
from ..utils.file_structure_manager import FileStructManager


class DataProcessor:
    """
    Class, that get data from data_conveyor and process it:
    1) Train or predict data
    2) Provide monitoring (showing metrics to console and tesorboard)
    """

    class LearningRate:
        """
        Learning rate manage strategy.
        This class provide lr decay by loss values. If loss doesn't update minimum throw defined number of steps - lr decay to defined coefficient
        """

        def __init__(self, config: {}, is_continue: bool = False):
            """
            :param config: learning rate config
            """
            self.__value = float(config['learning_rate']['start_value'])
            self.__decrease_coefficient = float(config['learning_rate']['decrease_coefficient'])
            self.__steps_before_decrease = config['learning_rate']['steps_before_decrease']
            if is_continue and 'first_steps_before_decrease' in config['learning_rate']:
                self.__decrease_after_first_steps_num = config['learning_rate']['first_steps_before_decrease']
                self.__first_decrease_coefficient = config['learning_rate']['first_decrease_coefficient']
            self.__cur_step = 0
            self.__min_loss = None
            self.__just_decreased = False

        def value(self, cur_loss: float = None) -> float:
            """
            Get value of current leraning rate
            :param cur_loss: current loss value
            :return: learning rate value
            """
            self.__just_decreased = False

            if hasattr(self, "_LearningRate__decrease_after_first_steps_num") and self.__cur_step == self.__decrease_after_first_steps_num:
                self.set_value(self.__value / self.__first_decrease_coefficient)

            if cur_loss is None:
                self.__cur_step += 1
                return self.__value

            if self.__min_loss is None:
                self.__min_loss = cur_loss

            if cur_loss < self.__min_loss:
                print("LR: Clear steps num")
                self.__cur_step = 0
                self.__min_loss = cur_loss

            if self.__cur_step > 0 and (self.__cur_step % self.__steps_before_decrease) == 0:
                self.__value /= self.__decrease_coefficient
                self.__min_loss = None
                print('Decrease lr to', self.__value)
                self.__just_decreased = True
                self.__cur_step = 0
                return self.__value

            self.__cur_step += 1
            return self.__value

        def set_value(self, value):
            self.__value = value
            self.__cur_step = 0
            self.__min_loss = None

        def lr_just_decreased(self) -> bool:
            return self.__just_decreased

    def __init__(self, model, train_pipeline: TrainPipeline, config: {}, file_struct_manager: FileStructManager, is_cuda=True, for_train: bool = True):
        """
        :param model: model object
        :param config: data processor conifig
        :param file_struct_manager: file stucture manager
        """

        self.__is_cuda = is_cuda
        self.__file_struct_manager = file_struct_manager

        self.__model = Model(model, file_struct_manager)

        if for_train:
            self.metrics_processor = train_pipeline.get_metrics_procesor()
            self.__criterion = train_pipeline.get_loss()

            if self.__is_cuda:
                self.__criterion.to('cuda:0')

            self.__learning_rate = self.LearningRate(config, is_continue=config['start_from'] == 'continue')

            self.__optimizer_fnc = getattr(torch.optim, config['optimizer'])
            if config['optimizer'] == "Adam":
                self.__optimizer = self.__optimizer_fnc(params=self.__model.model().parameters(), weight_decay=0.0005, eps=10e-10, lr=self.__learning_rate.value())
            else:
                self.__optimizer = self.__optimizer_fnc(params=self.__model.model().parameters(), weight_decay=0.0005, lr=self.__learning_rate.value())

            self.__epoch_num = 0

        if self.__is_cuda:
            self.__model.to_cuda()

        if config['start_from'] == 'continue':
            state_manager = StateManager(file_struct_manager, preffix="best")
            state_manager.unpack()
            self.load_weights(state_manager.get_files()['weights_file'])
            if for_train:
                self.load_state(state_manager.get_files()['state_file'])
            state_manager.clear_files()

    def model(self):
        """
        Get model
        """
        return self.__model.model()

    def learning_rate(self) -> LearningRate:
        return self.__learning_rate

    def predict(self, data, is_train=False, prev_pose: np.array = None) -> np.array:
        """
        Make predict by data
        :param prev_pose: previous position
        :param data: data in dict
        :param is_train: is data processor need train on data or just predict
        :return: processed output
        """

        def make_predict(prev_pose=None):
            if prev_pose is None:
                prev_pose = torch.cat([data['target']['prev']['pose'], data['target']['prev']['orientation']], dim=1)
            else:
                prev_pose = torch.from_numpy(prev_pose)

            if self.__is_cuda:
                dict_recursive_bypass(data['data'], lambda v: v.to('cuda:0'))
                prev_pose = prev_pose.to('cuda:0')
            return self.__model({'prev_img': data['data']['prev_img'], 'cur_img': data['data']['cur_img'], 'prev_pos': prev_pose})

        if is_train:
            self.__model.model().train()
            output = make_predict(prev_pose)
        else:
            with torch.no_grad():
                output = make_predict(prev_pose)
            self.__model.model().eval()

        return output

    def process_batch(self, batch: {}, is_train) -> None:
        """
        Process one batch of data
        :param batch: dict, contains data and target keys
        :param is_train: is data processor need to train on data or validate
        """
        if self.__is_cuda:
            dict_recursive_bypass(batch['target'], lambda v: v.to('cuda:0'))

        if is_train:
            self.__optimizer.zero_grad()

        self.__optimizer.zero_grad()

        res = self.predict(batch, is_train)
        self.metrics_processor.calc_metrics(res, batch['target'], is_train)

        loss = self.__criterion(res, batch['target'])
        if is_train:
            loss.backward()
            self.__optimizer.step()

        self.metrics_processor.set_loss(loss.data[0], is_train)

    def train_epoch(self, train_dataloader, validation_dataloader, epoch_idx: int, msg: str = None) -> {}:
        """
        Train one epoch
        :param train_dataloader: dataloader with train dataset
        :param validation_dataloader: dataloader with validation dataset
        :param epoch_idx: index of epoch
        :return: dict of events
        """
        for batch in tqdm(train_dataloader, desc="train" if msg is None else "train_" + msg, leave=False):
            self.process_batch(batch, is_train=True)
        for batch in tqdm(validation_dataloader, desc="validation" if msg is None else "validation_" + msg, leave=False):
            self.process_batch(batch, is_train=False)

        self.__epoch_num = epoch_idx

    def update_lr(self, lr: float) -> None:
        """
        Provide learning rate decay for optimizer
        :param lr: target learning rate
        """
        for param_group in self.__optimizer.param_groups:
            param_group['lr'] = lr

    def get_state(self) -> {}:
        """
        Get model and optimizer state dicts
        """
        return {'weights': self.__model.model().state_dict(), 'optimizer': self.__optimizer.state_dict()}

    def get_last_epoch_idx(self):
        return self.__epoch_num

    def load_state(self, optimizer_state_path: str, with_cur_lr: bool=False) -> None:
        """
        Load state of darta processor. Model state load separately
        :param with_cur_lr: is need to load learning rate
        :param optimizer_state_path: path to optimizer state path
        """
        print("Data processor inited by file: ", optimizer_state_path, end='; ')
        state = torch.load(optimizer_state_path)
        print('state dict len before:', len(state), end='; ')
        state = {k: v for k, v in state.items() if k in self.__optimizer.state_dict()}
        print('state dict len after:', len(state), end='; ')
        self.__optimizer.load_state_dict(state)
        print('done')

        with open(self.__file_struct_manager.data_processor_state_file(), 'r') as in_file:
            dp_state = json.load(in_file)
            self.__epoch_num = dp_state['last_epoch_idx']
            if not with_cur_lr:
                self.update_lr(dp_state['lr'])
                self.__learning_rate.set_value(dp_state['lr'])

    def load_weights(self, path):
        self.__model.load_weights(path)

    def save_weights(self):
        self.__model.save_weights()

    def save_state(self):
        """
        Save state of optimizer and perform epochs number
        :return:
        """
        torch.save(self.__optimizer.state_dict(), self.__file_struct_manager.optimizer_state_file())

        with open(self.__file_struct_manager.data_processor_state_file(), 'w') as out:
            json.dump({"last_epoch_idx": self.__epoch_num, 'lr': self.__learning_rate.value()}, out)
