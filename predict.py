from random import randint

import cv2
import json
import os
from multiprocessing import freeze_support

import torch
import numpy as np
from tqdm import tqdm

from data_conveyor.augmentations import Resize, CentralCrop, ToPyTorch
from data_conveyor.data_conveyor import Dataset
from data_processor import DataProcessor
from data_processor.state_manager import StateManager


def main():
    with open(os.path.join("workdir", "config.json"), 'r') as file:
        config = json.load(file)

    # batch_size = int(config['data_conveyor']['batch_size'])
    # threads_num = int(config['data_conveyor']['threads_num'])

    # test_loader = torch.utils.data.DataLoader(
    #     Dataset('test', config),
    #     batch_size=batch_size, shuffle=True,
    #     num_workers=threads_num, pin_memory=True)
    #
    state_manager = StateManager(None, config)

    data_processor = state_manager.load(config)

    dir = os.path.join(config['workdir_path'], 'test')
    images = [{'path': os.path.join(dir, im), 'id': int(im.split('.')[0])} for im in os.listdir(dir)]

    indices = [i['id'] for i in images]

    augmentations_config = config['data_conveyor']['test']['const_augmentations']
    resize = Resize(augmentations_config)
    ccrop = CentralCrop(augmentations_config)
    to_pytorch = ToPyTorch()

    with open('result.csv', 'w') as res:
        res.write("id,predicted\n")

        for im in tqdm(images, desc="predict", leave=False):
            data = to_pytorch(ccrop(resize(cv2.imread(im['path'])))).unsqueeze_(0)
            [_, preds], _ = data_processor.predict(data)
            res.write("{},{}\n".format(im['id'], int(preds)))
            res.flush()

        for i in range(1, 12800):
            if i not in indices:
                res.write("{},{}\n".format(i, randint(1, 128)))
                res.flush()

    data_processor.close()


if __name__ == "__main__":
    freeze_support()
    main()
