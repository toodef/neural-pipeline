import torch
from tqdm import tqdm

from neural_pipeline.data_processor.model import Model
from neural_pipeline.train_pipeline.train_pipeline import AbstractTrainPipeline


class DataProcessor:
    """
    Class, that get data from data_producer and process it:
    1) Train or predict data
    2) Provide monitoring (showing metrics to console and tesorboard)
    """

    class LearningRate:
        """
        Learning rate manage strategy.
        This class provide lr decay by loss values. If loss doesn't update minimum throw defined number of steps - lr decay to defined coefficient
        """
        def __init__(self, config: {}):
            """
            :param config: learning rate config
            """
            self.__value = float(config['learning_rate']['start_value'])
            self.__decrease_coefficient = float(config['learning_rate']['decrease_coefficient'])
            self.__steps_before_decrease = config['learning_rate']['steps_before_decrease']
            self.__cur_step = 0
            self.__min_loss = None
            self.__just_decreased = False

        def value(self, cur_loss: float = None) -> float:
            """
            Get value of current leraning rate
            :param cur_loss: current loss value
            :return: learning rate value
            """
            if cur_loss is None:
                return self.__value

            self.__just_decreased = False
            if self.__min_loss is None:
                self.__min_loss = cur_loss

            if cur_loss < self.__min_loss:
                print("LR: Clear steps num")
                self.__cur_step = 0
                self.__min_loss = cur_loss

            if self.__cur_step > 0 and (self.__cur_step % self.__steps_before_decrease) == 0:
                self.__value /= self.__decrease_coefficient
                print('Decrease lr to', self.__value)
                self.__just_decreased = True

            self.__cur_step += 1
            return self.__value

        def lr_just_decreased(self) -> bool:
            return self.__just_decreased

    def __init__(self, model: Model, train_pipeline: AbstractTrainPipeline, for_train: bool):
        """
        :param model: model object
        :param train_pipeline: train pipeline object
        :param for_train: is data processor created for train model
        """
        self.__model = model
        self.__train_pipeline = train_pipeline
        self.__is_train = for_train

        if for_train:
            self.__model.model().train()
        else:
            self.__model.model().eval()

        self.__optimizer = None
        self.__loss = None

    def set_optimizer(self, optimizer: torch.optim) -> object:
        """
        Set optimizer
        :param optimizer: optimizer object
        """
        self.__optimizer = optimizer
        return self

    def set_loss(self, loss: torch.nn.Function) -> object:
        """
        Set optimizer
        :param loss: loss object
        """
        self.__loss = loss
        return self

    def predict(self, data: torch.Tensor, is_train: bool = False) -> object:
        """
        Make predict by data
        :param data: input for model
        :return: processed output
        """
        if is_train:
            with torch.no_grad():
                return self.__model(data)
        return self.__model(data)

    def process_batch(self, batch: {}) -> None:
        """
        Process one batch odf data
        :param batch: dta for process
        """
        self.__optimizer.zero_grad()

        output = self.predict(batch['data'])

        loss = self.__loss(output, batch['target'])
        loss.backward()
        self.__optimizer.step()

        self.__train_pipeline.metrics_processor().calculate_metrics(output, batch['target'])

    def train_epoch(self, train_dataloader, validation_dataloader) -> None:
        """
        Train one epoch
        :param train_dataloader: dataloader with train dataset
        :param validation_dataloader: dataloader with validation dataset
        """
        for batch in tqdm(train_dataloader, desc="train", leave=False):
            self.process_batch(batch)
        for batch in tqdm(validation_dataloader, desc="validation", leave=False):
            self.process_batch(batch)

    def __update_lr(self, lr) -> None:
        """
        Provide learning rate decay for optimizer
        :param lr: target learning rate
        """
        for param_group in self.__optimizer.param_groups:
            param_group['lr'] = lr

    def get_state(self) -> {}:
        """
        Get model and optimizer state dicts
        """
        pass

    def load_state(self, optimizer_state_path: str) -> None:
        """
        Load state of darta processor. Model state load separately
        :param optimizer_state_path: path to optimizer state path
        """
        pass

    def save_state(self):
        """
        Save state of optimizer and perform epochs number
        :return:
        """
        pass
