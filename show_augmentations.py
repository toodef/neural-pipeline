from multiprocessing import freeze_support

import cv2
import json
import os

import numpy as np
import torch

from data_conveyor.data_conveyor import Dataset

with open(os.path.join("workdir", "config.json"), 'r') as file:
    config = json.load(file)

loader = torch.utils.data.DataLoader(
    Dataset('validation', config),
    batch_size=1, shuffle=False,
    num_workers=1, pin_memory=True)

if __name__ == "__main__":
    freeze_support()
    for data in loader:
        img = data['data'].numpy()[0]
        img = np.moveaxis(img, 0, -1)
        cv2.imshow('img', img)
        cv2.waitKey()
