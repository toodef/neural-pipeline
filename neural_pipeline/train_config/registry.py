from abc import ABCMeta, abstractmethod

from torch import optim
from torch import nn

__all__ = ['RegestryEntry']


class RegestryEntry(metaclass=ABCMeta):
    def __init__(self, instance_type: type):
        self._type = instance_type
        self._params = None
        self._instance = None

    @abstractmethod
    def get_params(self) -> {}:
        """
        Get dict of parameters values

        :return: dict of parameters with names in keys
        """

    def load_params(self, params: {}) -> 'RegestryEntry':
        """
        Init object by parameters

        :param params: dict of parameters
        :return: instance of object
        """
        self._params = params.copy()
        self._instance = self._init_by_params(params)
        return self

    def get_instance(self) -> object:
        return self._instance

    @abstractmethod
    def _init_by_params(self, params: {}) -> object:
        """
        Init object by parameters

        :param params: dict of parameters
        :return: instance of object
        """


class AdamEntry(RegestryEntry):
    _param_names = ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad']

    def __init__(self):
        super().__init__(optim.Adam)

    def get_params(self):
        res = {}
        for param_name in self._param_names:
            if param_name in self._param_names:
                res[param_name] = self._params[param_name]
        return res

    def _init_by_params(self, params: {}) -> optim.Adam:
        return optim.Adam(params=params['params'], lr=params['lr'] if 'lr' in params else 1e-3,
                          betas=params['betas'] if 'betas' in params else (0.9, 0.999), eps=params['eps'] if 'eps' in params else 1e-8,
                          weight_decay=params['weight_decay'] if 'weight_decay' in params else 0,
                          amsgrad=params['amsgrad'] if 'amsgrad' in params else False)


class BCELossEntry(RegestryEntry):
    def __init__(self):
        super().__init__(nn.BCELoss)

    def get_params(self) -> {}:
        return {}

    def _init_by_params(self, params: {}) -> object:
        return nn.BCELoss()


registry = {type(optim.Adam): AdamEntry(), type(nn.BCELoss): BCELossEntry()}
