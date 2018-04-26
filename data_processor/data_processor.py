import time

import torch
from tqdm import tqdm

from data_processor.model import Model
from data_processor.monitoring import Monitor
from utils.config import InitedByConfig


class DataProcessor(InitedByConfig):
    def __init__(self, config: {}):
        self.__is_cuda = True
        self.__model = torch.nn.DataParallel(Model(config).model())
        if self.__is_cuda:
            self.__model = self.__model.cuda()
        self.__learning_rate = float(config['network']['learning_rate'])
        self.__optimizer = getattr(torch.optim, config['network']['optimizer'])(self.__model.parameters(), lr=self.__learning_rate, weight_decay=1.e-4)
        self.__criterion = torch.nn.CrossEntropyLoss()
        if self.__is_cuda:
            self.__criterion = self.__criterion.cuda()
        self.__monitor = Monitor(config)
        self.clear_metrics()
        self.__batch_size = int(config['data_conveyor']['batch_size'])

    def process_batch(self, input, target, is_train):
        self.__model.train(is_train)

        # target = target.cuda(async=True)
        if self.__is_cuda:
            input = input.cuda()
            target = target.cuda()

        input_var = torch.autograd.Variable(input, volatile=not is_train)
        target_var = torch.autograd.Variable(target, volatile=not is_train)

        output = self.__model(input_var)
        _, preds = torch.max(output.data, 1)

        if is_train:
            loss = self.__criterion(output, target_var)
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

            self.__metrics['loss'] += loss.data[0] * input_var.size(0)
            self.__metrics['train_accuracy'] += torch.sum(preds == target_var.data)
        else:
            self.__metrics['val_accuracy'] += torch.sum(preds == target_var.data)

        self.__images_processeed['train' if is_train else 'val'] += self.__batch_size

    def train_epoch(self, train_dataloader, validation_dataloader, epoch_idx: int):
        start_time = time.time()

        for batch in tqdm(train_dataloader, desc="train", leave=False):
            self.process_batch(batch['data'], batch['target'], is_train=True)
        for batch in tqdm(validation_dataloader, desc="validation", leave=False):
            self.process_batch(batch['data'], batch['target'], is_train=False)

        cur_metrics = self.get_metrics()
        self.__monitor.update(epoch_idx, cur_metrics)
        print("Epoch: {}; loss: {}; val_accuracy: {}; train_accuracy: {}, elapsed {} min"
              .format(epoch_idx + 1, cur_metrics['loss'], cur_metrics['val_accuracy'], cur_metrics['train_accuracy'], (time.time() - start_time) // 60))
        self.clear_metrics()

    def get_metrics(self):
        return {"loss": self.__metrics['loss'] / self.__images_processeed['train'],
                "val_accuracy": self.__metrics['val_accuracy'] / self.__images_processeed['val'],
                "train_accuracy": self.__metrics['train_accuracy'] / self.__images_processeed['train']}

    def clear_metrics(self):
        self.__metrics = {"loss": 0, "val_accuracy": 0, "train_accuracy": 0}
        self.__images_processeed = {"val": 0, "train": 0}

    def save_state(self, path: str):
        pass

    def close(self):
        self.__monitor.close()

    def _required_params(self):
        return {
                "network": {
                  "optimiser": ["Adam", "SGD"],
                  "learning_rate": "Learning rate value",
                },
                "workdir_path": "workdir"
              }
