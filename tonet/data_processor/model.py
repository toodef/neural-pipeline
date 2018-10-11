import torch
import torchvision
from torch.utils import model_zoo

from neural_pipeline.tonet.data_processor import u_net_model
from neural_pipeline.tonet.utils.config import InitedByConfig
from neural_pipeline.tonet.utils.file_structure_manager import FileStructManager

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth'
}

start_modes = ['begin', 'url', 'continue']
model_types = ['classifier', 'u_net']


class Model(InitedByConfig):
    """
    Model is a neural network architecture. This class provide initialisation, call and serialisation for it
    """

    class ModelException(Exception):
        def __init__(self, message):
            super(Model.ModelException, self).__init__(message)
            self.__message = message

        def __str__(self):
            return self.__message

    def __init__(self, config: {}, file_struct_manager: FileStructManager, classes_num: int, is_cuda: bool = True):
        """
        :param config: model config
        :param file_struct_manager: file structure manager
        :param classes_num: number of classes
        :param is_cuda: is need to run model on cuda device
        """
        super().__init__()

        self.__file_struct_manager = file_struct_manager

        self.__config = config
        self.__classes_num = classes_num

        channels_num = self.__config['data_size'][2]

        if self.__config["model_type"] == "classifier":
            self.__model = getattr(torchvision.models, self.__config['architecture'])()

            if config['start_from'] == 'url':
                self.__load_weights_by_url()

            self.__model.classifier = torch.nn.Linear(self.__model.classifier.in_features, self.__classes_num)
        elif self.__config['model_type'] == "u_net":
            if config['start_from'] == 'url':
                self.__model = getattr(u_net_model, self.__config['architecture'])(classes_num=classes_num, in_channels=channels_num, weights_url=model_urls[self.__config['architecture']])
            else:
                self.__model = getattr(u_net_model, self.__config['architecture'])(classes_num=classes_num, in_channels=channels_num)

        self.__model = torch.nn.DataParallel(self.__model)
        if is_cuda:
            self.__model = self.__model.cuda()

    def model(self) -> torch.nn.Module:
        return self.__model

    def __load_weights_by_url(self) -> None:
        """
        Load pretrained weights from url
        """
        print("Model weights inited by url")
        model_url = model_urls[self.__config['architecture']]

        pretrained_weights = model_zoo.load_url(model_url)
        model_state_dict = self.__model.state_dict()
        pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_state_dict}
        self.__model.load_state_dict(pretrained_weights)

    def load_weights(self, weights_file: str) -> None:
        """
        Load weight from file
        :param weights_file: path to weights file
        """
        print("Model inited by file: ", weights_file)

        pretrained_weights = torch.load(weights_file)
        pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in self.__model.state_dict()}
        self.__model.load_state_dict(pretrained_weights)

    def save_weights(self):
        """
        Serialize weights to file
        :return:
        """
        torch.save(self.__model.state_dict(), self.__file_struct_manager.weights_file())

    def __config_start_mode(self):
        """
        Get start mode from config
        """
        return self.__config['start_from']

    def _required_params(self):
        return {"data_processor": {
            "architecture": ["resnet34"],
            "weights_dir": ["weights"],
            "start_from": ["begin", "url"]
        },
            "workdir_path": "workdir"
        }

    def __call__(self, x):
        """
        Call torch.nn.Module __call__ method
        :param x: data
        """
        return self.__model(x)
