import torch

from data_processor.model_initializer import Model
from utils.config import InitedByConfig


class ImageProcessor(InitedByConfig):
    def __init__(self, config: {}):
        self.__model = torch.nn.DataParallel(Model(config)).cuda()
        self.__learning_rate = float(config['network']['learning_rate'])
        self.__optimizer = getattr(torch.optim, config['network']['optimizer'])(self.__model.parameters(), lr=self.__learning_rate, weight_decay=1.e-4)
        self.__criterion = torch.nn.CrossEntropyLoss().cuda()

    def train_batch(self, input, target):
        self.__model.train()

        target = target.cuda(assync=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = self.__model(input_var)
        loss = self.__criterion(output, target_var)

        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()

    def get_loss_value(self, images: [{}]):
        pass

    def get_accuracy(self, images: [{}]):
        pass

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
