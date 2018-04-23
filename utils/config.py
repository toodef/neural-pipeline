from abc import ABCMeta, abstractmethod


class ConfigException(Exception):
    def __init__(self, class_name):
        super(ConfigException, self).__init__()
        self.__wrong_params = []
        self.__missed_params = []
        self.__class_name = class_name

    def wrong_params(self, params: []):
        self.__wrong_params += params
        return self

    def missed_params(self, params: []):
        self.__missed_params += params
        return self

    def __str__(self):
        message = "Config contains wrong params for class: " + self.__class_name

        if str(self.__missed_params):
            message += "  Doesn't contains params: " + str(self.__missed_params)
        if str(self.__wrong_params):
            message += "  Wrong params: " + str(self.__wrong_params)


class InitedByConfig(metaclass=ABCMeta):
    def check_config(self, config: {}):
        """
        Check config is correct for this module
        :return: None
        :raise: ConfigException if config is wrong
        """
        needed_params = self._check_config()

        missed_params = []

        for p in needed_params.keys():
            if not p in config:
                missed_params.append(p)

        if len(missed_params) == 0:
            return

        raise ConfigException(type(self).__name__).missed_params(missed_params)

    @abstractmethod
    def _required_params(self):
        """
        Method, that return required parameters in config for inherit class
        :return: required parameters as dict with same structure as config dict
        """
