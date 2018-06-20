import json
import os
from multiprocessing import freeze_support

import torch

from data_conveyor.data_conveyor import Dataset
from data_processor import DataProcessor
from data_processor.state_manager import StateManager


def main():
    with open(os.path.join(r"C:\workspace\nn_projects\furniture_segmentation", "default_config.ns"), 'r') as file:
        config = json.load(file)

    epoch_num = int(config['data_conveyor']['epoch_num'])

    batch_size = int(config['data_conveyor']['batch_size'])
    threads_num = int(config['data_conveyor']['threads_num'])

    with open(os.path.join(r"C:\workspace\nn_projects\furniture_segmentation\workdir", "validation.json"), 'r') as file:
        validation_pathes = json.load(file)
    with open(os.path.join(r"C:\workspace\nn_projects\furniture_segmentation\workdir", "train.json"), 'r') as file:
        train_pathes = json.load(file)

    train_loader = torch.utils.data.DataLoader(
        Dataset(config['data_conveyor']['train'], train_pathes),
        batch_size=batch_size, shuffle=True,
        num_workers=threads_num, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        Dataset(config['data_conveyor']['train'], validation_pathes),
        batch_size=batch_size, shuffle=False,
        num_workers=threads_num, pin_memory=True)

    data_processor = DataProcessor(config['data_processor'])
    state_manager = StateManager(data_processor, config)

    for epoch_idx in range(epoch_num):
        data_processor.train_epoch(train_loader, val_loader, epoch_idx)
        state_manager.save()

    data_processor.close()


if __name__ == "__main__":
    freeze_support()
    main()
