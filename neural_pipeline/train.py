from neural_pipeline.data_processor import Model, TrainDataProcessor
from neural_pipeline.data_processor.monitoring import Monitor
from neural_pipeline.utils.file_structure_manager import FileStructManager
from neural_pipeline.train_config.train_config import TrainConfig
from neural_pipeline.data_processor.state_manager import StateManager


class Trainer:
    """
    Class, that provide model training
    """

    def __init__(self, model: Model, train_config: TrainConfig, file_struct_manager: FileStructManager):
        self.__train_config = train_config
        self.__file_struct_manager = file_struct_manager
        self.__model = model

        self.__is_cuda = True
        self.__epoch_num = 100
        self.__need_resume = False

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

    def train(self) -> None:
        """
        Train model
        """
        data_processor = TrainDataProcessor(self.__model, self.__train_config, self.__file_struct_manager, is_cuda=True)
        state_manager = StateManager(self.__file_struct_manager)

        if self.__need_resume:
            state_manager.unpack()
            data_processor.load()
            state_manager.pack()

        start_epoch_idx = data_processor.get_last_epoch_idx() + 1 if data_processor.get_last_epoch_idx() > 0 else 0

        monitor = Monitor(self.__file_struct_manager, False, start_epoch_idx, self.__train_config.experiment_name())
        for epoch_idx in range(start_epoch_idx, self.__epoch_num + start_epoch_idx):
            for stage in self.__train_config.stages():
                stage.run(data_processor)

                data_processor.save_state()
                state_manager.pack()

                self._update_monitor(monitor, data_processor, epoch_idx)
                self._reset_metrics(data_processor)

    def _reset_metrics(self, data_processor: TrainDataProcessor) -> None:
        """
        Reset metrics. This method called after every epoch
        :param data_processor: data processor, that train model
        """
        data_processor.reset_losses()
        self.__iterate_by_stages(lambda stage: stage.metrics_processor().reset_metrics())

    def _update_monitor(self, monitor: Monitor, data_processor: TrainDataProcessor, epoch_idx: int) -> None:
        """
        Update monitor. This method call after every epoch
        :param monitor: monitor
        :param data_processor: data processor, that train model
        :param epoch_idx: index of epoch, that was ended
        """
        monitor.update_losses(epoch_idx, data_processor.get_losses())

    def __iterate_by_stages(self, func: callable):
        for stage in self.__train_config.stages():
            func(stage)