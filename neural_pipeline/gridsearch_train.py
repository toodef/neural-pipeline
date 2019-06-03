import os
import numpy as np
import torch

from neural_pipeline import AbstractMetric
from neural_pipeline.train_config.train_config import ComparableTrainConfig
from neural_pipeline.utils.fsm import MultipleFSM


class GridSearchTrainer:
    class MetricValAggregator:
        def __init__(self, metric: AbstractMetric, method: str = 'min'):
            self._values = []
            self._metric = metric

            self._process_vals = None

            if method == 'min':
                self._process_vals = self._calc_min
            elif len(method) > 12 and method[:12] == 'calc_around_':
                self._process_vals = lambda: self._calc_around_min(int(method[:12]))
            else:
                raise NotImplementedError("Methods for process metric must be 'min' or 'calc_around_N' where N is integer positive number")

        def update(self):
            self._values.append(np.mean(self._metric.get_values()))

        def _calc_min(self) -> float:
            return self._values[np.argmin(self._values)]

        def _calc_around_min(self, num_around: int) -> float:
            min_idx = np.argmin(self._values)
            num_back = min_idx - num_around
            if num_back < 0:
                num_back = 0
            num_forward = min_idx + num_around
            if num_forward > len(self._values) - 1:
                num_forward = len(self._values) - 1
            return np.mean(self._values[num_back: num_forward])

        def get_val(self) -> float:
            return self._process_vals()

    def __init__(self, train_configs: [ComparableTrainConfig], workdir: str, device: torch.device = None, is_continue: bool = False):
        self._train_configs = train_configs
        self._device = device

        self._workdir = workdir
        self._state = {}

        self._fsm = MultipleFSM(self._workdir, is_continue=is_continue)
        self._init_monitor_clbks = []

        self._epoch_num = 100

        if is_continue:
            self._load_state()

    def _load_state(self) -> None:
        """
        Internal method for gridsearch state load

        :return: self object
        """
        with open(self.__state_file_path(), 'r') as file:
            self._state = json.load(file)

    def train(self):
        if os.path.exists(self._workdir) or not os.path.isdir(self._workdir):
            os.makedirs(self._workdir)

        with open(self.__state_file_path(), 'w') as file:
            for i, comparable_train_config in enumerate(self._train_configs):
                train_config_name = str(i)

                if train_config_name in self._state:
                    continue

                print("Train '{}'".format(train_config_name))

                self._fsm.set_namespace(train_config_name)
                trainer = Trainer(comparable_train_config.get_train_config(), self._fsm, device=self._device)

                for init_monitor in self._init_monitor_clbks:
                    trainer.monitor_hub.add_monitor(init_monitor())
                metric_aggregator = self.MetricValAggregator(comparable_train_config.get_metric_for_compare())
                trainer.add_on_epoch_end_callback(metric_aggregator.update)
                trainer.set_epoch_num(self._epoch_num)
                trainer.train()

                cur_state = {'canceled': True, 'params': comparable_train_config.get_params(), 'metric_val': metric_aggregator.get_val()}

                self._state[train_config_name] = cur_state
                json.dump(self._state, file)

    def __state_file_path(self) -> str:
        """
        Internam method for compile state file path

        :return: path
        """
        return os.path.join(self._workdir, 'gridsearch_trainer.json')

    def set_epoch_num(self, epoch_num: int) -> 'GridSearchTrainer':
        self._epoch_num = epoch_num
        return self

    def add_init_monitor_clbk(self, init_monitor_clbk: callable) -> 'GridSearchTrainer':
        self._init_monitor_clbks.append(init_monitor_clbk)
        return self

    def fsm(self) -> 'FileStructManager':
        return self._fsm

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        best_params = None
        cur_best_metric = None
        for exp_name, state in self._state.items():
            if cur_best_metric is None or cur_best_metric < state['metric_val']:
                cur_best_metric = state['metric_val']
                best_params = state['params']

        print("Best parameters:", best_params)
        print("Best metric value:", cur_best_metric)
