from tensorboardX import SummaryWriter


class Monitor:
    def __init__(self, images_num: int):
        dir = "workdir\\logs"
        self.__writer = SummaryWriter(dir)
        self.__images_num = images_num

    def update(self, epoch: int, data_processor: DataProcessor):
        self.__update_tensordoard(epoch, data_processor)

    def __update_tensordoard(self, epoch: int, data_processor: DataProcessor):
        self.__writer.add_scalar('train\\loss', data_processor.get_loss_value(self.__images_num), global_step=epoch)
        self.__writer.add_scalar('train\\accuracy', data_processor.get_accuracy(self.__images_num), global_step=epoch)

    def close(self):
        self.__writer.close()
