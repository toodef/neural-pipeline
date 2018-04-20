import os
import requests

import torch
import torchvision.models as models


class ModelWorker:
    def __init__(self, config: {}):
        self.__model = None
        self.__config = config

    def model(self):
        return self.__model

    def __init_from_config(self):
        self.__model = getattr(models, self.__config['network']['name'])()

        start_mode = self.__config_start_mode()
        if start_mode == "begin":
            return

        if start_mode == "url":
            model_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
            output_file = model_url.split("/")[-1]
            if not os.path.isfile(output_file):
                response = requests.get(model_url)
                with open(output_file, 'wb') as input:
                    input.write(response.content)

            pretrained_weights = torch.load(output_file)

    def __config_start_mode(self):
        return self.__config['network']['start_from']
