class DecayingLearningRate:
    """
    Learning rate manage strategy.
    This class provide lr decay by loss values. If loss doesn't update minimum throw defined number of steps - lr decay to defined coefficient
    """

    def __init__(self, start_value: float, decay_coefficient: float, steps_before_decay: int):
        """
        :param start_value: start value
        :param decay_coefficient: coefficient of decaying
        :param steps_before_decay: steps before decay
        """
        self.__value = start_value
        self.__decay_coefficient = decay_coefficient
        self.__steps_before_decrease = steps_before_decay
        self.__cur_step = 0
        self.__min_loss = None
        self.__just_decreased = False

    def value(self, cur_loss: float = None) -> float:
        """
        Get value of current leraning rate
        :param cur_loss: current loss value
        :return: learning rate value
        """
        self.__just_decreased = False

        if cur_loss is None:
            self.__cur_step += 1
            return self.__value

        if self.__min_loss is None:
            self.__min_loss = cur_loss

        if cur_loss < self.__min_loss:
            print("LR: Clear steps num")
            self.__cur_step = 0
            self.__min_loss = cur_loss

        if self.__cur_step > 0 and (self.__cur_step % self.__steps_before_decrease) == 0:
            self.__value /= self.__decay_coefficient
            self.__min_loss = None
            print('Decrease lr to', self.__value)
            self.__just_decreased = True
            self.__cur_step = 0
            return self.__value

        self.__cur_step += 1
        return self.__value

    def set_value(self, value):
        self.__value = value
        self.__cur_step = 0
        self.__min_loss = None

    def lr_just_decreased(self) -> bool:
        return self.__just_decreased

