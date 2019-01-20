import torch
from torch.nn import Module

from neural_pipeline.utils.file_structure_manager import FileStructManager

__all__ = ['Model']


class Model:
    """
    Wrapper for :class:`torch.nn.Module`. This class provide initialization, call and serialization for it

    :param base_model: :class:`torch.nn.Module` object
    :param file_struct_manager: file structure manager
    """
    def __init__(self, base_model: Module, file_struct_manager: FileStructManager):
        self._base_model = base_model
        self._fsm = file_struct_manager

    def model(self) -> Module:
        """
        Get internal :class:`torch.nn.Module` object

        :return: internal :class:`torch.nn.Module` object
        """
        return self._base_model

    def load_weights(self) -> None:
        """
        Load weight from checkpoint
        """
        weights_file = self._fsm.weights_file()
        print("Model inited by file: ", weights_file, end='; ')
        pretrained_weights = torch.load(weights_file)
        print("weights before:", len(pretrained_weights), end='; ')
        pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in self._base_model.state_dict()}
        self._base_model.load_state_dict(pretrained_weights)
        print("weights after:", len(pretrained_weights))

    def save_weights(self) -> None:
        """
        Serialize weights to file
        """
        torch.save(self._base_model.state_dict(), self._fsm.weights_file())

    def __call__(self, x):
        """
        Call torch.nn.Module __call__ method

        :param x: model input data
        """
        return self._base_model(x)

    def to_cuda(self) -> None:
        """
        Pass model to CUDA device
        """
        self._base_model.to('cuda')
