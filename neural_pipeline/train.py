from neural_pipeline.data_processor import Model
from neural_pipeline.data_processor.monitoring import Monitor
from neural_pipeline.utils.file_structure_manager import FileStructManager
from neural_pipeline.data_processor.data_processor import DataProcessor
from neural_pipeline.train_config.train_config import TrainConfig
from neural_pipeline.data_processor.state_manager import StateManager
from neural_pipeline.data_producer.data_producer import DataProducer


class Trainer:
    def __init__(self, model: Model, train_config: TrainConfig, file_struct_manager: FileStructManager, train_producer: DataProducer, validation_producer: DataProducer=None):
        self.__train_config = train_config
        self.__file_struct_manager = file_struct_manager
        self.__train_producer = train_producer
        self.__validation_producer = validation_producer
        self.__model = model

        self.__is_cuda = True
        self.__epoch_num = 2000

    def set_epoch_num(self, epoch_number: int) -> 'Trainer':
        self.__epoch_num = epoch_number
        return self

    def train(self):
        train_loader = self.__train_producer.get_loader()
        val_loader = self.__validation_producer.get_loader()

        data_processor = DataProcessor(self.__model, self.__train_config, self.__file_struct_manager, is_cuda=True, for_train=True)
        state_manager = StateManager(self.__file_struct_manager)

        start_epoch_idx = data_processor.get_last_epoch_idx() + 1 if data_processor.get_last_epoch_idx() > 0 else 0

        monitor = Monitor(self.__file_struct_manager, False, start_epoch_idx, self.__train_config.experiment_name())
        for epoch_idx in range(start_epoch_idx, self.__epoch_num + start_epoch_idx):
            data_processor.train_epoch(train_loader, val_loader, epoch_idx)

            data_processor.save_state()
            state_manager.pack()

            monitor.update_metrics(epoch_idx, self.__train_config.metrics_processor().get_metrics())
            monitor.update_losses(epoch_idx, data_processor.get_losses())

            data_processor.reset_losses()
            self.__train_config.metrics_processor().reset_metrics()
