import json
import os
from multiprocessing import freeze_support

import torch

from data_conveyor.data_conveyor import Dataset
from data_processor import DataProcessor
from data_processor.state_manager import StateManager


def main():
    with open(os.path.join("workdir", "config.json"), 'r') as file:
        config = json.load(file)

    batch_size = int(config['data_conveyor']['batch_size'])
    threads_num = int(config['data_conveyor']['threads_num'])

    train_loader = torch.utils.data.DataLoader(
        Dataset('train', config),
        batch_size=batch_size, shuffle=True,
        num_workers=threads_num, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        Dataset('validation', config),
        batch_size=batch_size, shuffle=False,
        num_workers=threads_num, pin_memory=True)

    # data_processor = DataProcessor(config)
    state_manager = StateManager(None, config)

    data_processor = state_manager.load(config)
    data_processor.train_epoch(train_loader, val_loader, 0)

    data_processor.close()


if __name__ == "__main__":
    freeze_support()
    main()
