import json
import numpy as np

import torch
from torch.nn import Module

from neural_pipeline.data_processor.model import Model
from neural_pipeline.utils.file_structure_manager import FileStructManager
from neural_pipeline.utils.utils import dict_recursive_bypass

__all__ = ['DataProcessor', 'TrainDataProcessor']


class DataProcessor:
    """
    DataProcessor manage: model, data processing, device choosing

    :param model: model, that will be used for process data
    :param file_struct_manager: file structure manager
    :param is_cuda: is processing will be in CUDA device
    """
    def __init__(self, model: Module, file_struct_manager: FileStructManager, is_cuda=True):
        self._is_cuda = is_cuda
        self._file_struct_manager = file_struct_manager

        self._model = Model(model, file_struct_manager)

        if self._is_cuda:
            self.model().cuda()

    def model(self) -> Module:
        """
        Get current module
        """
        return self._model.model()

    def predict(self, data) -> object:
        """
        Make predict by data

        :param data: data in dict with key `data`
        :return: processed output
        :rtype: the model output type
        """

        def make_predict():
            if self._is_cuda:
                data['data'] = self._pass_data_to_device(data['data'])
            return self._model(data['data'])

        self.model().eval()
        with torch.no_grad():
            output = make_predict()
        return output

    def load(self) -> None:
        """
        Load model weights from checkpoint
        """
        self._model.load_weights()

    @staticmethod
    def _pass_data_to_device(data: torch.Tensor or dict) -> torch.Tensor or dict:
        """
        Internal method, that pass data to specified device
        :param data: data as dict or torch.Tensor
        :return: processed on target device
        """
        if isinstance(data, dict):
            return dict_recursive_bypass(data, lambda v: v.to('cuda:0'))
        else:
            return data.to('cuda:0')


class TrainDataProcessor(DataProcessor):
    """
    TrainDataProcessor is make all of DataProcessor but produce training process.

    :param model: model, that will be used for process data
    :param train_config: train config
    :param file_struct_manager: file structure manager
    :param is_cuda: is processing will be in CUDA device
    """

    def __init__(self, model: Module, train_config: 'TrainConfig', file_struct_manager: FileStructManager, is_cuda=True):
        super().__init__(model, file_struct_manager, is_cuda)

        self.__criterion = train_config.loss()

        if self._is_cuda:
            self.__criterion.to('cuda:0')

        self.__optimizer = train_config.optimizer()

        self.__epoch_num = 0

    def predict(self, data, is_train=False) -> torch.Tensor or dict:
        """
        Make predict by data. If ``is_train`` was ``True``

        :param data: data in dict
        :param is_train: is data processor need train on data or just predict
        :return: processed output
        :rtype: model return type
        """

        def make_predict():
            if self._is_cuda:
                data['data'] = self._pass_data_to_device(data['data'])
            return self._model(data['data'])

        if is_train:
            self.model().train()
            output = make_predict()
        else:
            output = super().predict(data)

        return output

    def process_batch(self, batch: {}, is_train: bool, metrics_processor: 'AbstractMetricsProcessor' = None) -> np.ndarray:
        """
        Process one batch of data

        :param batch: dict, contains 'data' and 'target' keys. The values for key must be instance of torch.Tensor or dict
        :param is_train: is batch process for train
        :param metrics_processor: metrics processor for collect metrics after batch is processed
        :return: array of losses with shape (N, ...) where N is batch size
        """
        if self._is_cuda:
            batch['target'] = self._pass_data_to_device(batch['target'])

        if is_train:
            self.__optimizer.zero_grad()
        res = self.predict(batch, is_train)

        if metrics_processor is not None:
            metrics_processor.calc_metrics(res, batch['target'])

        loss = self.__criterion(res, batch['target'])
        if is_train:
            loss.backward()
            self.__optimizer.step()

        return loss.data.cpu().numpy()

    def update_lr(self, lr: float) -> None:
        """
        Update learning rate straight to optimizer

        :param lr: target learning rate
        """
        for param_group in self.__optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """
        Get learning rate from optimizer
        """
        for param_group in self.__optimizer.param_groups:
            return param_group['lr']

    def get_last_epoch_idx(self):
        return self.__epoch_num

    def get_state(self) -> {}:
        """
        Get model and optimizer state dicts

        :return: dict with keys [weights, optimizer]
        """
        return {'weights': self._model.model().state_dict(), 'optimizer': self.__optimizer.state_dict()}

    def load(self) -> None:
        """
        Load state of model, optimizer and TrainDataProcessor from checkpoint
        """
        super().load()

        print("Data processor inited by file:", self._file_struct_manager.optimizer_state_file(), end='; ')
        state = torch.load(self._file_struct_manager.optimizer_state_file())
        print('state dict len before:', len(state), end='; ')
        state = {k: v for k, v in state.items() if k in self.__optimizer.state_dict()}
        print('state dict len after:', len(state), end='; ')
        self.__optimizer.load_state_dict(state)
        print('done')

        with open(self._file_struct_manager.data_processor_state_file(), 'r') as in_file:
            dp_state = json.load(in_file)
            self.__epoch_num = dp_state['last_epoch_idx']

    def save_state(self) -> None:
        """
        Save state of optimizer and perform epochs number
        """
        torch.save(self.__optimizer.state_dict(), self._file_struct_manager.optimizer_state_file())

        with open(self._file_struct_manager.data_processor_state_file(), 'w') as out:
            json.dump({"last_epoch_idx": self.__epoch_num}, out)

        self._model.save_weights()
