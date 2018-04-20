import os
import requests

import torch

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np


class ImageProcessor:
    class ImageProcessorException(Exception):
        def __init__(self, message: str):
            self.__msg = message

        def __str__(self):
            return self.__msg

    def __init__(self, classes_number: int, train_images_num: int, image_size: [], batch_size, epoch_every_train_num: int, threads_num: int=0):
        if type(image_size) != list or len(image_size) != 3:
            raise self.ImageProcessorException("Bad image size data. This must be list of 3 integers")
        self.__image_size = image_size
        self.__classes_num = classes_number
        self.__on_epoch = None
        self.__init_nn()
        self.__learning_rate = 0.1
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=self.__learning_rate, weight_decay=1.e-4)
        self.__train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root='train', transform=transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])),
            batch_size=batch_size, shuffle=True,
            num_workers=threads_num, pin_memory=True)

        self.__iteration_idx = 0
        self.__train_images_num = train_images_num
        # self.__saver = tf.train.Saver()
        self.__epoch_every_train_num = epoch_every_train_num

    def __init_nn(self):
        def load_by_url(url: str, dst: str):
            response = requests.get(url)
            with open(dst, 'wb') as input:
                input.write(response.content)

        model_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
        output_file = model_url.split("/")[-1]
        if not os.path.isfile(output_file):
            load_by_url(model_url, output_file)
        pretrained_weights = torch.load(output_file)
        self.__model = models.resnet34(pretrained=True)
        self.__model.load_state_dict(pretrained_weights, strict=True)

    def __init_label(self, label_id: int):
        label = np.zeros(self.__classes_num)
        label[label_id] = 1.0
        return label

    def train_batch(self, images: [{}]):
        pass

    def set_on_epoch(self, callback: callable):
        self.__on_epoch = callback

    def get_loss_value(self, images: [{}]):
        pass

    def get_accuracy(self, images: [{}]):
        pass

    def get_cur_epoch(self):
        pass

    def save_state(self, path: str):
        pass


class Predictor:
    def __init__(self, path, classes_number: int, image_size: []):
        pass

    def predict(self, image):
        pass
