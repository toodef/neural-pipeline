import time
from functools import partial

import torch
from tqdm import tqdm

from data_processor.model import Model
from data_processor.monitoring import Monitor
from utils.config import InitedByConfig


class DataProcessor(InitedByConfig):
    def __init__(self, config: {}):
        self.__is_cuda = True
        self.__model = Model(config).model()
        if self.__is_cuda:
            self.__model = self.__model.cuda()
        self.__learning_rate = float(config['network']['learning_rate'])

        self.__optimizer_fnc = getattr(torch.optim, config['network']['optimizer'])
        self.__optimizer = self.__optimizer_fnc(params=self.__model.parameters(), weight_decay=1.e-4, lr=self.__learning_rate)
        self.__decrease_lr_every_epoch = config['network']['decrease_lr_every_epoch']

        self.__criterion = torch.nn.CrossEntropyLoss()
        if self.__is_cuda:
            self.__criterion = self.__criterion.cuda()
        self.__monitor = Monitor(config)
        self.clear_metrics()
        self.__batch_size = int(config['data_conveyor']['batch_size'])

        self.__epoch_num = 0

    def predict(self, input, is_train=False):
        if self.__is_cuda:
            input = input.cuda(async=is_train)

        if is_train:
            self.__model.train()
        else:
            self.__model.eval()

        input_var = torch.autograd.Variable(input, volatile=not is_train)
        output = self.__model(input_var)
        return torch.max(output.data, 1), output

    def process_batch(self, input, target, is_train):
        self.__model.train(is_train)

        if self.__is_cuda:
            target = target.cuda(async=True)

        inputs_num = input.size(0)
        target_var = torch.autograd.Variable(target, volatile=not is_train)

        if is_train:
            self.__optimizer.zero_grad()

        [_, preds], output = self.predict(input, is_train)

        if is_train:
            loss = self.__criterion(output, target_var)
            loss.backward()
            self.__metrics['loss'] += loss.data[0] * inputs_num
            self.__metrics['train_accuracy'] += torch.sum(preds == target_var.data)

            # torch.nn.utils.clip_grad_norm(self.__model.parameters(), 1 / 128.)
            self.__optimizer.step()
        else:
            loss = self.__criterion(output, target_var)
            self.__metrics['val_loss'] += loss.data[0] * inputs_num
            self.__metrics['val_accuracy'] += torch.sum(preds == target_var.data)

        self.__images_processeed['train' if is_train else 'val'] += self.__batch_size

    def train_epoch(self, train_dataloader, validation_dataloader, epoch_idx: int):
        if epoch_idx == 1:
            self.__learning_rate /= 10
            self.__optimizer = self.__optimizer_fnc(params=self.__model.parameters(), weight_decay=1.e-4, lr=self.__learning_rate)

        if epoch_idx > 1 and self.__decrease_lr_every_epoch % epoch_idx == 0:
            self.__learning_rate /= 2
            self.__optimizer = self.__optimizer_fnc(params=self.__model.parameters(), weight_decay=1.e-4, lr=self.__learning_rate)

        for batch in tqdm(train_dataloader, desc="train", leave=False):
            self.process_batch(batch['data'], batch['target'], is_train=True)
        for batch in tqdm(validation_dataloader, desc="validation", leave=False):
            self.process_batch(batch['data'], batch['target'], is_train=False)

        cur_metrics = self.get_metrics()
        self.__monitor.update(epoch_idx, cur_metrics)
        self.clear_metrics()

    def get_metrics(self):
        val_acc = self.__metrics['val_accuracy'] / self.__images_processeed['val']
        train_acc = self.__metrics['train_accuracy'] / self.__images_processeed['train']
        return {"loss": self.__metrics['loss'] / self.__images_processeed['train'],
                "val_loss": self.__metrics['val_loss'] / self.__images_processeed['train'],
                "val_accuracy": val_acc,
                "train_accuracy": train_acc,
                "train_min_val_acc": train_acc - val_acc}

    def clear_metrics(self):
        self.__metrics = {"loss": 0, "val_loss": 0, "val_accuracy": 0, "train_accuracy": 0, "train_min_val_acc": 0}
        self.__images_processeed = {"val": 0, "train": 0}

    def get_state(self):
        return {'weights': self.__model.state_dict(), 'optimizer': self.__optimizer.state_dict()}

    def load_state(self, optimizer_state: str):
        state = torch.load(optimizer_state)
        state = {k: v for k, v in state.items() if k in self.__optimizer.state_dict()}
        self.__optimizer.load_state_dict(state)

    def load_weights(self, path):
        self.__model.load_weights(path)

    def save_weights(self, path):
        torch.save(self.__model.state_dict(), path)
        # self.__model.save_weights(path)

    def save_state(self, path):
        torch.save(self.__optimizer.state_dict(), path)

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
