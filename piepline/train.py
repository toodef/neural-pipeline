"""
The main module for training process
"""
import json
import os
import numpy as np
import torch
from torch.nn import Module

from piepline import AbstractMetric

from piepline.data_processor import TrainDataProcessor
from piepline.utils import FileStructManager, CheckpointsManager
from piepline.train_config.train_config import TrainConfig, ComparableTrainConfig
from piepline.monitoring import MonitorHub, ConsoleMonitor
from piepline.utils.fsm import MultipleFSM

__all__ = ['Trainer']


class LearningRate:
    """
    Basic learning rate class
    """

    def __init__(self, value: float):
        self._value = value

    def value(self) -> float:
        """
        Get value of current learning rate

        :return: current value
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
    This class provide lr decaying by defined metric value (by :arg:`target_value_clbk`).
    If metric value doesn't update minimum after defined number of steps (:arg:`patience`) - lr was decaying
    by defined coefficient (:arg:`decay_coefficient`).

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
    Class, that run drive process.

    Trainer get list of training stages and every epoch loop over it.

    Training process looks like:

    .. highlight:: python
    .. code-block:: python

        for epoch in epochs_num:
            for stage in training_stages:
                stage.run()
                monitor_hub.update_metrics(stage.metrics_processor().get_metrics())
            save_state()
            on_epoch_end_callback()

    :param train_config: :class:`TrainConfig` object
    :param fsm: :class:`FileStructManager` object
    :param device: device for training process
    """

    class TrainerException(Exception):
        def __init__(self, msg):
            super().__init__()
            self._msg = msg

        def __str__(self):
            return self._msg

    def __init__(self, train_config: TrainConfig, fsm: FileStructManager, device: torch.device = None):
        self._fsm = fsm
        self.monitor_hub = MonitorHub()

        self._checkpoint_manager = CheckpointsManager(self._fsm)

        self.__epoch_num = 100
        self._resume_from = None
        self._on_epoch_end = []
        self._best_state_rule = None

        self._train_config = train_config
        self._data_processor = TrainDataProcessor(self._train_config, device).set_checkpoints_manager(self._checkpoint_manager)
        self._lr = LearningRate(self._data_processor.get_lr())

        self._stop_rules = []

    def set_epoch_num(self, epoch_number: int) -> 'Trainer':
        """
        Define number of epoch for training. One epoch - one iteration over all train stages

        :param epoch_number: number of training epoch
        :return: self object
        """
        self.__epoch_num = epoch_number
        return self

    def resume(self, from_best_checkpoint: bool) -> 'Trainer':
        """
        Resume train from last checkpoint

        :param from_best_checkpoint: is need to continue from best checkpoint
        :return: self object
        """
        self._resume_from = 'last' if from_best_checkpoint is False else 'best'
        return self

    def enable_lr_decaying(self, coeff: float, patience: int, target_val_clbk: callable) -> 'Trainer':
        """
        Enable rearing rate decaying. Learning rate decay when `target_val_clbk` returns doesn't update
        minimum for `patience` steps

        :param coeff: lr decay coefficient
        :param patience: number of steps
        :param target_val_clbk: callback which returns the value that is used for lr decaying
        :return: self object
        """
        self._lr = DecayingLR(self._data_processor.get_lr(), coeff, patience, target_val_clbk)
        return self

    def train(self) -> None:
        """
        Run training process
        """
        if len(self._train_config.stages()) < 1:
            raise self.TrainerException("There's no sages for training")

        best_checkpoints_manager = None
        cur_best_state = None
        if self._best_state_rule is not None:
            best_checkpoints_manager = CheckpointsManager(self._fsm, 'best')

        start_epoch_idx = 1
        if self._resume_from is not None:
            start_epoch_idx += self._resume()
        self.monitor_hub.add_monitor(ConsoleMonitor())

        with self.monitor_hub:
            for epoch_idx in range(start_epoch_idx, self.__epoch_num + start_epoch_idx):
                if True in [stop_rule() for stop_rule in self._stop_rules]:
                    break

                self.monitor_hub.set_epoch_num(epoch_idx)
                for stage in self._train_config.stages():
                    stage.run(self._data_processor)

                    if stage.metrics_processor() is not None:
                        self.monitor_hub.update_metrics(stage.metrics_processor().get_metrics())

                new_best_state = self._save_state(self._checkpoint_manager, best_checkpoints_manager, cur_best_state, epoch_idx)
                if new_best_state is not None:
                    cur_best_state = new_best_state

                self._data_processor.update_lr(self._lr.value())

                for clbk in self._on_epoch_end:
                    clbk()

                self._update_losses()
                self.__iterate_by_stages(lambda s: s.on_epoch_end())

    def _resume(self) -> int:
        if self._resume_from == 'last':
            ckpts_manager = self._checkpoint_manager
        elif self._checkpoint_manager == 'best':
            ckpts_manager = CheckpointsManager(self._fsm, 'best')
        else:
            raise NotImplementedError("Resume parameter may be only 'last' or 'best' not {}".format(self._resume_from))
        ckpts_manager.unpack()
        self._data_processor.load()

        with open(ckpts_manager.trainer_file(), 'r') as file:
            start_epoch_idx = json.load(file)['last_epoch'] + 1

        ckpts_manager.pack()
        return start_epoch_idx

    def _save_state(self, ckpts_manager: CheckpointsManager, best_ckpts_manager: CheckpointsManager or None,
                    cur_best_state: float or None, epoch_idx: int) -> float or None:
        """
        Internal method used for save states after epoch end

        :param ckpts_manager: ordinal checkpoints manager
        :param best_ckpts_manager: checkpoints manager, used for store best stages
        :param cur_best_state: current best stage metric value
        :return: new best stage metric value or None if it not update
        """
        def save_trainer(ckp_manager):
            with open(ckp_manager.trainer_file(), 'w') as out:
                json.dump({'last_epoch': epoch_idx}, out)

        if self._best_state_rule is not None:
            new_best_state = self._best_state_rule()
            if cur_best_state is None:
                self._data_processor.save_state()
                save_trainer(ckpts_manager)
                ckpts_manager.pack()
                return new_best_state
            else:
                if new_best_state <= cur_best_state:
                    self._data_processor.set_checkpoints_manager(best_ckpts_manager)
                    self._data_processor.save_state()
                    save_trainer(best_ckpts_manager)
                    best_ckpts_manager.pack()
                    self._data_processor.set_checkpoints_manager(ckpts_manager)
                    return new_best_state

        self._data_processor.save_state()
        save_trainer(ckpts_manager)
        ckpts_manager.pack()
        return None

    def _update_losses(self) -> None:
        """
        Update loses procedure
        """
        losses = {}
        for stage in self._train_config.stages():
            if stage.get_losses() is not None:
                losses[stage.name()] = stage.get_losses()
        self.monitor_hub.update_losses(losses)

    def data_processor(self) -> TrainDataProcessor:
        """
        Get data processor object

        :return: data processor
        """
        return self._data_processor

    def enable_best_states_saving(self, rule: callable) -> 'Trainer':
        """
        Enable best states saving

        Best stages will save when return of `rule` update minimum

        :param rule: callback which returns the value that is used for define when need store best metric
        :return: self object
        """
        self._best_state_rule = rule
        return self

    def disable_best_states_saving(self) -> 'Trainer':
        """
        Enable best states saving

        :return: self object
        """
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

    def add_stop_rule(self, rule: callable) -> 'Trainer':
        """
        Add the rule that control training process interruption

        Params:
            rule (callable): callable, that doesn't get params and return boolean. When one of rules returns `True` training loop will be interrupted

        Returns:
            self object

        Examples:

        .. highlight:: python
        .. code-block:: python

            trainer.add_stop_rule(lambda: trainer.data_processor().get_lr() < 1e-6)
        """
        self._stop_rules.append(rule)
        return self

    def train_config(self) -> TrainConfig:
        """
        Get train config

        :return: TrainConfig object
        """
        return self._train_config

    def __iterate_by_stages(self, func: callable) -> None:
        """
        Internal method, that used for iterate by stages

        :param func: callback, that calls for every stage
        """
        for stage in self._train_config.stages():
            func(stage)
