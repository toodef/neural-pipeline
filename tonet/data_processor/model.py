import os

import requests

import torch
import torchvision

from tonet.tonet.data_processor import u_net_model
from tonet.tonet.utils.config import InitedByConfig
from tonet.tonet.utils.file_structure_manager import FileStructManager

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
    class ModelException(Exception):
        def __init__(self, message):
            super(Model.ModelException, self).__init__(message)
            self.__message = message

        def __str__(self):
            return self.__message

    def __init__(self, config: {}, file_struct_manager: FileStructManager, classes_num: int):
        super().__init__()

        self.__file_struct_manager = file_struct_manager

        self.__config = config
        self.__classes_num = classes_num

        channels_num = self.__config['data_size'][2]

        if self.__config["model_type"] == "classifier":
            self.__model = getattr(torchvision.models, self.__config['architecture'])()
        elif self.__config['model_type'] == "u_net":
            self.__model = getattr(u_net_model, self.__config['architecture'])(classes_num=classes_num, in_channels=channels_num)

        self.__is_cuda = True
        if self.__is_cuda:
            self.__model = self.__model.cuda()

        self.__init_from_config()

    def model(self):
        return self.__model

    def __init_from_config(self):
        start_mode = self.__config_start_mode()
        if start_mode == start_modes[0]:
            return

        self.__weights_dir = self.__file_struct_manager.weights_dir()
        self.__weights_file = self.__file_struct_manager.weights_file()

    def __load_weights_by_url(self):
        model_url = model_urls[self.__config['architecture']]
        init_weights_file = os.path.join(self.__weights_dir, model_url.split("/")[-1])

        if not os.path.isfile(init_weights_file):
            if not os.path.exists(self.__weights_dir) or not os.path.isdir(self.__weights_dir):
                os.makedirs(self.__weights_dir)
            response = requests.get(model_url)
            with open(init_weights_file, 'wb') as file:
                file.write(response.content)

    def load_weights(self, weights_file: str, url=False):
        pretrained_weights = torch.load(weights_file)

        if not url:
            self.__model.classifier = torch.nn.Linear(self.__model.classifier.in_features, self.__classes_num)
            self.__model = torch.nn.DataParallel(self.__model)
            pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in self.__model.state_dict()}
            self.__model.load_state_dict(pretrained_weights)
        else:
            self.__load_weights_by_url()
            self.__model.load_state_dict(pretrained_weights)
            self.__model.classifier = torch.nn.Linear(self.__model.classifier.in_features, self.__classes_num)
            self.__model = torch.nn.DataParallel(self.__model)

    def save_weights(self):
        torch.save(self.__model.state_dict(), self.__weights_file)

    def __config_start_mode(self):
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
        return self.__model(x)
