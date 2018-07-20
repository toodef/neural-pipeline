import os


class FileStructManager:
    class FSMException(Exception):
        def __init__(self, message: str):
            self.__message = message

        def __str__(self):
            return self.__message

    def __init__(self, config_path: str):
        if not (os.path.exists(config_path) and os.path.isfile(config_path)):
            raise self.FSMException("Config path doesnt find [{}]".format(config_path))

        self.__config_dir = os.path.dirname(config_path)
        self.__data_dir = os.path.join(self.__config_dir, 'data')

        self.__create_data_folder()

    def conjfig_dir(self) -> str:
        return self.__config_dir

    def weights_dir(self) -> str:
        return self.__data_dir

    def weights_file(self, preffix: str=None) -> str:
        return os.path.join(self.weights_dir(), "{}_".format(preffix) if preffix is not None else "" + "weights.pth")

    def optimizer_state_dir(self) -> str:
        return self.__data_dir

    def optimizer_state_file(self, preffix: str=None) -> str:
        return os.path.join(self.optimizer_state_dir(), "{}_".format(preffix) if preffix is not None else "" + "state.pth")

    def __create_data_folder(self) -> None:
        if os.path.exists(self.__data_dir) and os.path.isdir(self.__data_dir):
            return
        os.mkdir(self.__data_dir)


class FileStructManagerNSProject(FileStructManager):
    def __init__(self, project_path: str, config_path: int):
        super().__init__(project_path)
