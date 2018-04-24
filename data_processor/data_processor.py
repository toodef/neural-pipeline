import torch

from data_processor.model import Model
from utils.config import InitedByConfig


class DataProcessor(InitedByConfig):
    def __init__(self, config: {}):
        self.__model = torch.nn.DataParallel(Model(config).model()).cuda()
        self.__learning_rate = float(config['network']['learning_rate'])
        self.__optimizer = getattr(torch.optim, config['network']['optimizer'])(self.__model.parameters(), lr=self.__learning_rate, weight_decay=1.e-4)
        self.__criterion = torch.nn.CrossEntropyLoss().cuda()
        self.__losses = 0
        self.__accuracies = 0

    def train_batch(self, input, target):
        self.__model.train(True)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())

        output = self.__model(input_var)
        _, preds = torch.max(output.data, 1)
        loss = self.__criterion(output, target_var)

        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()

        self.__losses += loss.data[0] * input_var.size(0)
        self.__accuracies += torch.sum(preds == target_var.data)

    def get_loss_value(self, images_num: int):
        return self.__losses / images_num

    def get_accuracy(self, images_num: int):
        return self.__accuracies / images_num

    def clear_metrics(self):
        self.__losses, self.__accuracies = 0, 0

    def get_cur_epoch(self):
        pass

    def save_state(self, path: str):
        pass

    def _required_params(self):
        return {
                "network": {
                  "optimiser": ["Adam", "SGD"],
                  "learning_rate": "Learning rate value",
                },
                "workdir_path": "workdir"
              }
