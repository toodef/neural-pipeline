import numpy as np

import torch
from piepline.utils import dict_recursive_bypass

from piepline.utils import CheckpointsManager
from torch.nn import Module

from piepline.data_processor.model import Model

__all__ = ['DataProcessor', 'TrainDataProcessor']


class DataProcessor:
    """
    DataProcessor manage: model, data processing, device choosing

    Args:
        model (Module): model, that will be used for process data
        device (torch.device): what device pass data for processing
    """

    def __init__(self, model: Module, device: torch.device = None):
        self._checkpoints_manager = None
        self._model = Model(model)
        self._device = device
        self._pick_model_input = lambda data: data['data']

    def set_checkpoints_manager(self, checkpoint_manager: CheckpointsManager) -> 'DataProcessor':
        self._checkpoints_manager = checkpoint_manager
        self._model.set_checkpoints_manager(checkpoint_manager)
        return self

    def model(self) -> Module:
        """
        Get current module
        """
        return self._model.model()

    def set_pick_model_input(self, pick_model_input: callable) -> 'DataProcessor':
        """
        Set callback, that will get output from :mod:`DataLoader` and return model input.

        Default mode:

        .. highlight:: python
        .. code-block:: python

        lambda data: data['data']

        Args:
            pick_model_input (callable): pick model input callable. This callback need to get one parameter: dataset output

        Returns:
            self object

        Examples:

        .. highlight:: python
        .. code-block:: python

            data_processor.set_pick_model_input(lambda data: data['data'])
            data_processor.set_pick_model_input(lambda data: data[0])
        """
        self._pick_model_input = pick_model_input
        return self

    def predict(self, data: torch.Tensor or dict) -> object:
        """
        Make predict by data

        :param data: data as :class:`torch.Tensor` or dict with key ``data``
        :return: processed output
        :rtype: the model output type
        """
        self.model().eval()
        with torch.no_grad():
            output = self._model(self._pick_model_input(data))
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

    def __init__(self, train_config: 'TrainConfig', device: torch.device = None):
        super().__init__(train_config.model(), device)

        self._data_preprocess = (lambda data: data) if device is None else self._pass_data_to_device
        self._pick_target = lambda data: data['target']

        self._loss_input_preproc = lambda data: data
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
            output = self._model(self._pick_model_input(data))
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
        internal_batch = self._data_preprocess(batch)

        if is_train:
            self.__optimizer.zero_grad()

        res = self.predict(internal_batch, is_train)
        loss = self.__criterion(res, self._pick_target(internal_batch))

        if is_train:
            loss.backward()
            self.__optimizer.step()

        with torch.no_grad():
            if metrics_processor is not None:
                metrics_processor.calc_metrics(res, self._pick_target(internal_batch))

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

    def set_pick_target(self, pick_target: callable) -> 'DataProcessor':
        """
        Set callback, that will get output from :mod:`DataLoader` and return target.

        Default mode:

        .. highlight:: python
        .. code-block:: python

        lambda data: data['target']

        Args:
            pick_target (callable): pick target callable. This callback need to get one parameter: dataset output

        Returns:
            self object

        Examples:

        .. highlight:: python
        .. code-block:: python

            data_processor.set_pick_target(lambda data: data['target'])
            data_processor.set_pick_target(lambda data: data[1])
        """
        self._pick_target = pick_target
        return self

    def set_data_preprocess(self, data_preprocess: callable) -> 'DataProcessor':
        """
        Set callback, that will get output from :mod:`DataLoader` and return preprocessed data.
        For example may be used for pass data to device.

        Default mode:

        .. highlight:: python
        .. code-block:: python

        :meth:`_pass_data_to_device`

        Args:
            data_preprocess (callable): preprocess callable. This callback need to get one parameter: dataset output

        Returns:
            self object

        Examples:

        .. highlight:: python
        .. code-block:: python

            from piepline.utils import dict_recursive_bypass
            data_processor.set_data_preprocess(lambda data: dict_recursive_bypass(data, lambda v: v.cuda()))
        """
        self._data_preprocess = data_preprocess
        return self

    def _pass_data_to_device(self, data) -> torch.Tensor or dict:
        """
        Internal method, that pass data to specified device
        :param data: data as any object type. If will passed to device if it's instance of :class:`torch.Tensor` or dict with key
        ``data``. Otherwise data will be doesn't changed
        :return: processed on target device
        """
        if isinstance(data, dict):
            return dict_recursive_bypass(data, lambda v: v.to(self._device))
        elif isinstance(data, torch.Tensor):
            return data.to(self._device)
        else:
            return data
