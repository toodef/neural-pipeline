import numpy as np

import torch

from neural_pipeline.utils import CheckpointsManager
from torch.nn import Module

from neural_pipeline.data_processor.model import Model

__all__ = ['DataProcessor', 'TrainDataProcessor']


class DataProcessor:
    """
    DataProcessor manage: model, data processing, device choosing

    :param model: model, that will be used for process data
    :param device: what device pass model and data for processing
    """

    def __init__(self, model: Module):
        self._checkpoints_manager = None

        self._model = Model(model)

    def set_checkpoints_manager(self, checkpoint_manager: CheckpointsManager) -> 'DataProcessor':
        self._checkpoints_manager = checkpoint_manager
        self._model.set_checkpoints_manager(checkpoint_manager)
        return self

    def model(self) -> Module:
        """
        Get current module
        """
        return self._model.model()

    def predict(self, data: torch.Tensor or dict) -> object:
        """
        Make predict by data

        :param data: data as :class:`torch.Tensor` or dict with key ``data``
        :return: processed output
        :rtype: the model output type
        """
        self.model().eval()
        with torch.no_grad():
            output = self._model(data['data'])
        return output

    def load(self) -> None:
        """
        Load model weights from checkpoint
        """
        self._model.load_weights()

    def save_state(self) -> None:
        """
        Save state of optimizer and perform epochs number
        """
        self._model.save_weights()


class TrainDataProcessor(DataProcessor):
    """
    TrainDataProcessor is make all of DataProcessor but produce training process.

    :param train_config: train config
    """

    class TDPException(Exception):
        def __init__(self, msg):
            self._msg = msg

        def __str__(self):
            return self._msg

    def __init__(self, train_config: 'TrainConfig'):
        super().__init__(train_config.model())

        self.__criterion = train_config.loss()
        self.__optimizer = train_config.optimizer()

    def predict(self, data, is_train=False) -> torch.Tensor or dict:
        """
        Make predict by data. If ``is_train`` is ``True`` - this operation will compute gradients. If
        ``is_train`` is ``False`` - this will work with ``model.eval()`` and ``torch.no_grad``

        :param data: data in dict
        :param is_train: is data processor need train on data or just predict
        :return: processed output
        :rtype: model return type
        """

        if is_train:
            self.model().train()
            output = self._model(data['data'])
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

    def get_state(self) -> {}:
        """
        Get model and optimizer state dicts

        :return: dict with keys [weights, optimizer]
        """
        return {'weights': self._model.model().state_dict(), 'optimizer': self.__optimizer.state_dict()}

    def _get_checkpoints_manager(self) -> CheckpointsManager:
        if self._checkpoints_manager is None:
            raise self.TDPException("Checkpoints manager doesn't specified. Use 'set_checkpoints_manager()'")
        return self._checkpoints_manager

    def load(self) -> None:
        """
        Load state of model, optimizer and TrainDataProcessor from checkpoint
        """
        super().load()
        cp_manager = self._get_checkpoints_manager()
        print("Optimizer inited by file:", cp_manager.optimizer_state_file(), end='; ')
        state = torch.load(cp_manager.optimizer_state_file())
        print('state dict len before:', len(state), end='; ')
        state = {k: v for k, v in state.items() if k in self.__optimizer.state_dict()}
        print('state dict len after:', len(state), end='; ')
        self.__optimizer.load_state_dict(state)
        print('done')

    def save_state(self) -> None:
        """
        Save state of optimizer and perform epochs number
        """
        super().save_state()
        torch.save(self.__optimizer.state_dict(), self._get_checkpoints_manager().optimizer_state_file())
