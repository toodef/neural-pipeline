import os
import sys


class FileStructManager:
    """
    This class manage data directory. It's get path to config and provide info about folder and interface for work with it
    """
    class FSMException(Exception):
        def __init__(self, message: str):
            self.__message = message

        def __str__(self):
            return self.__message

    def __init__(self, checkpoint_dir_path: str, logdir_path: str=None):
        """
        :param checkpoint_dir_path: path to directory with checkpoints
        :param logdir_path: logdir path. May be none if exists NN_LOGDIR environment variable. If nothig was defined - tensorboard will not work
        """

        if logdir_path is None:
            if 'NN_LOGDIR' in os.environ:
                logdir_path = os.environ['NN_LOGDIR']
            else:
                print("Logdir doesn't specified and NN_LOGDIR env variable also doesn't specified! Logs will not be writen!", file=sys.stderr)

        self.__logdir_path = logdir_path

        if not (os.path.exists(checkpoint_dir_path) and os.path.isfile(checkpoint_dir_path)):
            raise self.FSMException("Checkpoint directory doesn't find [{}]".format(checkpoint_dir_path))

        self.__checkpoint_dir = os.path.dirname(checkpoint_dir_path)
        self.__data_dir = os.path.join(self.__checkpoint_dir, 'data')

        self.__create_data_folder()

    def checkpoint_dir(self) -> str:
        """
        Get path of directory, contains config file
        """
        return self.__checkpoint_dir

    def weights_dir(self) -> str:
        """
        Get path of directory, contains mode weights file
        """
        return self.__data_dir

    def weights_file(self, preffix: str=None) -> str:
        """
        Get path of weights file
        """
        return os.path.join(self.weights_dir(), "{}_".format(preffix) if preffix is not None else "" + "weights.pth")

    def optimizer_state_dir(self) -> str:
        """
        Get path of directory, contains optimizer state file
        """
        return self.__data_dir

    def optimizer_state_file(self, preffix: str=None) -> str:
        """
        Get path of optimizer state file
        """
        return os.path.join(self.optimizer_state_dir(), "{}_".format(preffix) if preffix is not None else "" + "state.pth")

    def data_processor_state_file(self, preffix: str=None) -> str:
        """
        Get path of data processor state file
        """
        return os.path.join(self.optimizer_state_dir(), "{}_".format(preffix) if preffix is not None else "" + "data_processor_state.json")

    def logdir_path(self) -> str:
        """
        Get path of directory, there will be stored logs
        """
        return self.__logdir_path

    def __create_data_folder(self) -> None:
        if os.path.exists(self.__data_dir) and os.path.isdir(self.__data_dir):
            return
        os.mkdir(self.__data_dir)
