import json
import os
import time
from multiprocessing import freeze_support

import torch
from torchvision import transforms, datasets

from data_processor import DataProcessor


def main():
    with open(os.path.join("workdir", "config.json"), 'r') as file:
        config = json.load(file)

    epoch_num = int(config['data_conveyor']['epoch_num'])

    batch_size = int(config['data_conveyor']['batch_size'])
    threads_num = int(config['data_conveyor']['threads_num'])

    traindir = os.path.join(config['workdir_path'], 'train')
    valdir = os.path.join(config['workdir_path'], 'validation')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_folder = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    train_loader = torch.utils.data.DataLoader(
        train_folder,
        batch_size=batch_size, shuffle=True,
        num_workers=threads_num, pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         # transforms.Scale(256),
    #         # transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=batch_size, shuffle=False,
    #     num_workers=threads_num, pin_memory=True)

    data_processor = DataProcessor(config)

    images_num = len(train_folder)
    for epoch_idx in range(epoch_num):
        start_time = time.time()
        for (input, target) in train_loader:
            data_processor.train_batch(input, target)

        print("Epoch: {}; loss: {}; accuracy: {}; elapsed {} min"
              .format(epoch_idx + 1, data_processor.get_loss_value(images_num), data_processor.get_accuracy(images_num),
                      (time.time() - start_time) // 60))
        data_processor.clear_metrics()


if __name__ == "__main__":
    freeze_support()
    main()
