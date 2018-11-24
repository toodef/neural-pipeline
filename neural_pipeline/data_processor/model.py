import torch
from torch.nn import Module

from tonet.neural_pipeline.utils.file_structure_manager import FileStructManager


class Model:
    """
    Model is a neural network architecture. This class provide initialisation, call and serialisation for it
    """

    class ModelException(Exception):
        def __init__(self, message):
            super(Model.ModelException, self).__init__(message)
            self.__message = message

        def __str__(self):
            return self.__message

    def __init__(self, base_model: Module, file_struct_manager: FileStructManager):
        self.__base_model = base_model
        self.__file_struct_manager = file_struct_manager

    def model(self) -> Module:
        """
        Return torch.Module
        :return: module
        """
        return self.__base_model

    def load_weights(self, weights_file: str) -> None:
        """
        Load weight from file
        :param weights_file: path to weights file
        """
        print("Model inited by file: ", weights_file, end='; ')
        pretrained_weights = torch.load(weights_file)
        print("weights before: ", weights_file, end='; ')
        pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in self.__base_model.state_dict()}
        self.__base_model.load_state_dict(pretrained_weights)
        print("weights after: ", weights_file)

    def save_weights(self) -> None:
        """
        Serialize weights to file
        """
        torch.save(self.__base_model.state_dict(), self.__file_struct_manager.weights_file())

    def __call__(self, x):
        """
        Call torch.nn.Module __call__ method
        :param x: data
        """
        return self.__base_model(x)
