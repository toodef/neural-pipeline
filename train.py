import json
import os
from multiprocessing import freeze_support

import torch
from torchvision import transforms, datasets

from data_conveyor.data_conveyor import Dataset
from data_processor import DataProcessor
from data_processor.state_manager import StateManager


def main():
    with open(os.path.join("workdir", "config.json"), 'r') as file:
        config = json.load(file)

    epoch_num = int(config['data_conveyor']['epoch_num'])

    batch_size = int(config['data_conveyor']['batch_size'])
    threads_num = int(config['data_conveyor']['threads_num'])

    data_size = config['data_conveyor']['data_size']

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        Dataset('train', config, transforms.Compose([
            # transforms.Resize(size=(data_size[0], data_size[1])),
            transforms.RandomCrop(size=(data_size[0], data_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), percentage=1),
        batch_size=batch_size, shuffle=True,
        num_workers=threads_num, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        Dataset('validation', config, transforms.Compose([
            transforms.Resize(size=(data_size[0], data_size[1])),
            # transforms.CenterCrop(size=(data_size[0], data_size[1])),
            transforms.ToTensor(),
            normalize,
        ]), percentage=1),
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
