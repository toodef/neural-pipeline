import torch
from torch.nn import Module

from neural_pipeline.data_processor import TrainDataProcessor
from neural_pipeline.utils import FileStructManager, CheckpointsManager
from neural_pipeline.train_config.train_config import TrainConfig
from neural_pipeline.monitoring import MonitorHub, ConsoleMonitor
from neural_pipeline.utils.file_structure_manager import FolderRegistrable

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

    def __init__(self, model: Module, train_config: TrainConfig, file_struct_manager: FileStructManager,
                 device: torch.device = None):
        self._fsm = file_struct_manager
        self.monitor_hub = MonitorHub()

        self._checkpoint_manager = CheckpointsManager(self._fsm)

        self.__epoch_num = 100
        self.__need_resume = False
        self._on_epoch_end = []
        self._best_state_rule = None

        self.__train_config = train_config
        self._device = device
        self._data_processor = TrainDataProcessor(model, self.__train_config, self._device)\
            .set_checkpoints_manager(self._checkpoint_manager)
        self._lr = LearningRate(self._data_processor.get_lr())

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

        best_checkpoints_manager = None
        cur_best_state = None
        if self._best_state_rule is not None:
            best_checkpoints_manager = CheckpointsManager(self._fsm, 'best')

        if self.__need_resume:
            self._checkpoint_manager.unpack()
            self._data_processor.load()
            self._checkpoint_manager.pack()

        start_epoch_idx = 1

        self.monitor_hub.add_monitor(ConsoleMonitor())

        with self.monitor_hub:
            for epoch_idx in range(start_epoch_idx, self.__epoch_num + start_epoch_idx):
                self.monitor_hub.set_epoch_num(epoch_idx)
                for stage in self.__train_config.stages():
                    stage.run(self._data_processor)

                    if stage.metrics_processor() is not None:
                        self.monitor_hub.update_metrics(stage.metrics_processor().get_metrics())

                self._data_processor.save_state()

                new_best_state = self._save_state(self._checkpoint_manager, best_checkpoints_manager, cur_best_state)
                if new_best_state is not None:
                    cur_best_state = new_best_state

                self._data_processor.update_lr(self._lr.value())

                for clbk in self._on_epoch_end:
                    clbk()

                self._update_losses()
                self.__iterate_by_stages(lambda s: s.on_epoch_end())

    def _save_state(self, ckpts_manager: CheckpointsManager, best_ckpts_manager: CheckpointsManager or None,
                    cur_best_state: float or None) -> float or None:
        if self._best_state_rule is not None:
            new_best_state = self._best_state_rule()
            if cur_best_state is None:
                ckpts_manager.pack()
                return new_best_state
            else:
                if new_best_state <= cur_best_state:
                    best_ckpts_manager.pack()
                    return new_best_state

        ckpts_manager.pack()
        return None

    def _update_losses(self):
        losses = {}
        for stage in self.__train_config.stages():
            if stage.get_losses() is not None:
                losses[stage.name()] = stage.get_losses()
        self.monitor_hub.update_losses(losses)

    def data_processor(self) -> TrainDataProcessor:
        """
        Get data processor object

        :return: data processor
        """
        return self._data_processor

    def enable_best_states_storing(self, rule: callable) -> 'Trainer':
        self._best_state_rule = rule
        return self

    def disable_best_states_storing(self):
        self._best_state_rule = None
        return self

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
