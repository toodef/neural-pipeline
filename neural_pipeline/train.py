from torch.nn import Module

from neural_pipeline.data_processor import TrainDataProcessor
from neural_pipeline.utils.file_structure_manager import FileStructManager
from neural_pipeline.train_config.train_config import TrainConfig
from neural_pipeline.data_processor.state_manager import StateManager
from neural_pipeline.monitoring import MonitorHub, ConsoleMonitor

__all__ = ['Trainer', 'LearningRate', 'DecayingLR']


class LearningRate:
    """
    Basic learning rate class.
    """

    def __init__(self, value: float):
        self._value = value

    def value(self) -> float:
        """
        Get value of current learning rate
        """
        return self._value

    def set_value(self, value) -> None:
        """
        Set lr value

        :param value: lr value
        """
        self._value = value


class DecayingLR(LearningRate):
    """
    This class provide lr decaying by defined metric value (by :arg:`target_value_clbk`). If metric value doesn't update minimum throw defined number of steps -
    lr decay by defined coefficient.

    :param start_value: start value
    :param decay_coefficient: coefficient of decaying
    :param patience: steps before decay
    :param target_value_clbk: callable, that return target value for lr decaying
    """

    def __init__(self, start_value: float, decay_coefficient: float, patience: int, target_value_clbk: callable):
        super().__init__(start_value)

        self._decay_coefficient = decay_coefficient
        self._patience = patience
        self._cur_step = 1
        self._target_value_clbk = target_value_clbk
        self._cur_min_target_val = None

    def value(self) -> float:
        """
        Get value of current learning rate

        :return: learning rate value
        """
        metric_val = self._target_value_clbk()
        if metric_val is None:
            return self._value

        if self._cur_min_target_val is None:
            self._cur_min_target_val = metric_val

        if metric_val < self._cur_min_target_val:
            self._cur_step = 1
            self._cur_min_target_val = metric_val

        if self._cur_step > 0 and (self._cur_step % self._patience) == 0:
            self._value *= self._decay_coefficient
            self._cur_min_target_val = None
            self._cur_step = 1
            return self._value

        self._cur_step += 1
        return self._value

    def set_value(self, value):
        self._value = value
        self._cur_step = 0
        self._cur_min_target_val = None


class Trainer:
    """
    Class, that provide model training
    """

    class TrainerException(Exception):
        def __init__(self, msg):
            super().__init__()
            self._msg = msg

        def __str__(self):
            return self._msg

    def __init__(self, model: Module, train_config: TrainConfig, file_struct_manager: FileStructManager, is_cuda: bool = True):
        self.__train_config = train_config
        self.__file_struct_manager = file_struct_manager
        self.__model = model

        self.__is_cuda = is_cuda
        self.__epoch_num = 100
        self.__need_resume = False

        self.monitor_hub = MonitorHub()

        self._data_processor = TrainDataProcessor(self.__model, self.__train_config, self.__file_struct_manager,
                                                  is_cuda=self.__is_cuda)
        self._lr = LearningRate(self._data_processor.get_lr())

        self._on_epoch_end = []

    def set_epoch_num(self, epoch_number: int) -> 'Trainer':
        """
        Define number of training epoch
        :param epoch_number: number of training epoch
        :return: self object
        """
        self.__epoch_num = epoch_number
        return self

    def resume(self) -> 'Trainer':
        """
        Resume train from last checkpoint
        :return: self object
        """
        self.__need_resume = True
        return self

    def enable_lr_decaying(self, coeff: float, patience: int, target_val_clbk: callable) -> 'Trainer':
        self._lr = DecayingLR(self._data_processor.get_lr(), coeff, patience, target_val_clbk)
        return self

    def train(self) -> None:
        """
        Train model
        """
        if len(self.__train_config.stages()) < 1:
            raise self.TrainerException("There's no sages for training")

        state_manager = StateManager(self.__file_struct_manager)

        if self.__need_resume:
            state_manager.unpack()
            self._data_processor.load()
            state_manager.pack()

        start_epoch_idx = self._data_processor.get_last_epoch_idx() + 1 if self._data_processor.get_last_epoch_idx() > 0 else 0

        self.monitor_hub.add_monitor(ConsoleMonitor())

        with self.monitor_hub:
            for epoch_idx in range(start_epoch_idx, self.__epoch_num + start_epoch_idx):
                for stage in self.__train_config.stages():
                    stage.run(self._data_processor)

                    if stage.metrics_processor() is not None:
                        self.monitor_hub.update_metrics(epoch_idx, stage.metrics_processor().get_metrics())
                    if stage.get_losses() is not None:
                        self.monitor_hub.update_losses(epoch_idx, {stage.name(): stage.get_losses()})

                self._data_processor.save_state()
                state_manager.pack()

                self._data_processor.update_lr(self._lr.value())

                for clbk in self._on_epoch_end:
                    clbk()

                self.__iterate_by_stages(lambda s: s.on_epoch_end())

    def data_processor(self) -> TrainDataProcessor:
        """
        Get data processor object

        :return: data processor
        """
        return self._data_processor

    def add_on_epoch_end_callback(self, callback: callable) -> 'Trainer':
        """
        Add callback, that will be called after every epoch end

        :param callback: method, that will be called. This method may not get any parameters
        :return: self object
        """
        self._on_epoch_end.append(callback)
        return self

    def __iterate_by_stages(self, func: callable):
        for stage in self.__train_config.stages():
            func(stage)
