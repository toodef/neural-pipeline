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

    def __init__(self, checkpoint_dir_path: str, logdir_path: str=None, prefix: str = None):
        """
        :param checkpoint_dir_path: path to directory with checkpoints
        :param logdir_path: logdir path. May be none if exists NN_LOGDIR environment variable. If nothig was defined - tensorboard will not work
        :param prefix: prefix of stored files
        """

        if logdir_path is None:
            if 'NN_LOGDIR' in os.environ:
                logdir_path = os.environ['NN_LOGDIR']
            else:
                print("Logdir doesn't specified and NN_LOGDIR env variable also doesn't specified! Logs will not be writen!", file=sys.stderr)

        self.__logdir_path = logdir_path

        if not (os.path.exists(checkpoint_dir_path) and os.path.isdir(checkpoint_dir_path)):
            raise self.FSMException("Checkpoint directory doesn't find [{}]".format(checkpoint_dir_path))

        self.__checkpoint_dir = checkpoint_dir_path
        self.__prefix = prefix
        self.__create_logdir()

    def checkpoint_dir(self) -> str:
        """
        Get path of directory, contains config file
        """
        return self.__checkpoint_dir

    def weights_file(self) -> str:
        """
        Get path of weights file
        """
        return os.path.join(self.checkpoint_dir(), "{}_".format(self.__prefix) if self.__prefix is not None else "" + "weights.pth")

    def optimizer_state_dir(self) -> str:
        """
        Get path of directory, contains optimizer state file
        """
        return self.__checkpoint_dir

    def optimizer_state_file(self) -> str:
        """
        Get path of optimizer state file
        """
        return os.path.join(self.optimizer_state_dir(), "{}_".format(self.__prefix) if self.__prefix is not None else "" + "state.pth")

    def data_processor_state_file(self, preffix: str=None) -> str:
        """
        Get path of data processor state file
        """
        return os.path.join(self.optimizer_state_dir(), "{}_".format(preffix) if preffix is not None else "" + "dp_state.json")

    def logdir_path(self) -> str:
        """
        Get path of directory, there will be stored logs
        """
        return self.__logdir_path

    def __create_logdir(self) -> None:
        if self.__logdir_path is None or os.path.exists(self.__logdir_path) and os.path.isdir(self.__logdir_path):
            return
        os.mkdir(self.__logdir_path)
