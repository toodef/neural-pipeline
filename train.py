import json
import os
import time
from multiprocessing import freeze_support

import torch
from torchvision import transforms, datasets

from data_processor import DataProcessor
from data_processor.monitoring import Monitor


def main():
    with open(os.path.join("workdir", "config.json"), 'r') as file:
        config = json.load(file)

    epoch_num = int(config['data_conveyor']['epoch_num'])

    batch_size = int(config['data_conveyor']['batch_size'])
    threads_num = int(config['data_conveyor']['threads_num'])

    data_size = config['data_conveyor']['data_size']

    traindir = os.path.join(config['workdir_path'], 'train')
    valdir = os.path.join(config['workdir_path'], 'validation')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_folder = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.Resize(data_size[:2]),
        transforms.ToTensor(),
        normalize,
    ]))
    train_loader = torch.utils.data.DataLoader(
        train_folder,
        batch_size=batch_size, shuffle=True,
        num_workers=threads_num, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(data_size[:2]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=threads_num, pin_memory=True)

    data_processor = DataProcessor(config)

    for epoch_idx in range(epoch_num):
        data_processor.train_epoch(train_loader, val_loader, epoch_idx)
    data_processor.close()


if __name__ == "__main__":
    freeze_support()
    main()
