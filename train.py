import cv2
import json
import os
from multiprocessing import freeze_support

import torch

from data_conveyor.data_conveyor import Dataset
from data_processor import DataProcessor
from data_processor.state_manager import StateManager


class Blur(object):
    def __init__(self, params: () = (5, 5)):
        self.__params = params

    def __call__(self, image):
        cv2.blur(image, self.__params)


def main():
    with open(os.path.join("workdir", "config.json"), 'r') as file:
        config = json.load(file)

    epoch_num = int(config['data_conveyor']['epoch_num'])

    batch_size = int(config['data_conveyor']['batch_size'])
    threads_num = int(config['data_conveyor']['threads_num'])

    train_loader = torch.utils.data.DataLoader(
        Dataset('train', config, percentage=10),
        batch_size=batch_size, shuffle=True,
        num_workers=threads_num, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        Dataset('validation', config),
        batch_size=batch_size, shuffle=False,
        num_workers=threads_num, pin_memory=True)

    data_processor = DataProcessor(config)
    state_manager = StateManager(data_processor, config)

    for epoch_idx in range(epoch_num):
        data_processor.train_epoch(train_loader, val_loader, epoch_idx)
        state_manager.save()

    # state_manager.load(config)
    # data_processor.train_epoch(train_loader, val_loader, 0)

    data_processor.close()


if __name__ == "__main__":
    freeze_support()
    main()
