import os
from zipfile import ZipFile

__all__ = ['StateManager', 'FileStructManager']


class StateManager:
    """
    Class that manage states for DataProcessor files.

    All states pack to zip file. It contains few files: model weights, optimizer state, data processor state

    :param file_struct_manager: file structure manager
    :param prefix: prefix of saved and loaded files
    """

    class SMException(Exception):
        """
        Exception for :mod:`StateManager`
        """

        def __init__(self, message: str):
            self.__message = message

        def __str__(self):
            return self.__message

    def __init__(self, file_struct_manager: 'FileStructManager', prefix: str = None):
        self._file_struct_manager = file_struct_manager

        weights_file = self._file_struct_manager.weights_file()
        state_file = self._file_struct_manager.optimizer_state_file()

        checkpoints_dir = self._file_struct_manager.checkpoint_dir()
        if not (os.path.exists(checkpoints_dir) and os.path.isdir(checkpoints_dir)):
            raise self.SMException("Checkpoints dir doesn't exists: [{}]".format(checkpoints_dir))

        if os.path.exists(weights_file) and os.path.exists(state_file) and os.path.isfile(weights_file) and os.path.isfile(state_file):
            self.__preffix = "prev_start"
            self.pack()
            self.__preffix = None

        self.__preffix = prefix

    def unpack(self) -> None:
        """
        Unpack state files
        """
        result_file = self._construct_result_file()

        with ZipFile(result_file, 'r') as zipfile:
            zipfile.extractall(self._file_struct_manager.checkpoint_dir())

        weights_file = self._file_struct_manager.weights_file()
        state_file = self._file_struct_manager.optimizer_state_file()
        dp_state_file = self._file_struct_manager.data_processor_state_file()
        self._check_files([weights_file, state_file, dp_state_file])

    def clear_files(self) -> None:
        """
        Clear unpacked files
        """

        def rm_file(file: str):
            if os.path.exists(file) and os.path.isfile(file):
                os.remove(file)

        rm_file(self._file_struct_manager.weights_file())
        rm_file(self._file_struct_manager.optimizer_state_file())
        rm_file(self._file_struct_manager.data_processor_state_file())

    def pack(self) -> None:
        """
        Pack all files in zip
        """

        def rm_file(file: str):
            if os.path.exists(file) and os.path.isfile(file):
                os.remove(file)

        def rename_file(file: str):
            target = file + ".old"
            rm_file(target)
            if os.path.exists(file) and os.path.isfile(file):
                os.rename(file, target)

        weights_file = self._file_struct_manager.weights_file()
        state_file = self._file_struct_manager.optimizer_state_file()
        dp_state_file = self._file_struct_manager.data_processor_state_file()

        self._check_files([weights_file, state_file, dp_state_file])
        result_file = self._construct_result_file()

        rename_file(result_file)
        with ZipFile(result_file, 'w') as zipfile:
            zipfile.write(weights_file, os.path.basename(weights_file))
            zipfile.write(state_file, os.path.basename(state_file))
            zipfile.write(dp_state_file, os.path.basename(dp_state_file))

        self.clear_files()

    def _construct_result_file(self) -> str:
        """
        Internal method for compile result file name

        :return: path to result file
        """
        data_dir = self._file_struct_manager.checkpoint_dir()
        return os.path.join(data_dir, (self.__preffix + "_" if self.__preffix is not None else "") + "state.zip")

    def _check_files(self, files) -> None:
        """
        Internal method for checking files for condition of existing

        :param files: list of files pathes to check
        :raises: SMException
        """
        failed = []
        for f in files:
            if not (os.path.exists(f) and os.path.isfile(f)):
                failed.append(f)

        if len(failed) > 0:
            raise self.SMException("Some files doesn't exists: [{}]".format(';'.join(files)))


class FileStructManager:
    """
    This class manage data directory. It's get path to config and provide info about folder and interface for work with it

    :param checkpoint_dir_path: path to directory with checkpoints
    :param logdir_path: logdir path. May be none if exists NN_LOGDIR environment variable. If nothig was defined - tensorboard will not work
    :param prefix: prefix of stored files
    :param is_continue: is FileStructManager used for continue training or predict
    """

    class FSMException(Exception):
        def __init__(self, message: str):
            self.__message = message

        def __str__(self):
            return self.__message

    def __init__(self, checkpoint_dir_path: str, logdir_path: str, prefix: str = None, is_continue: bool = False):
        self.__logdir_path = logdir_path
        self.__checkpoint_dir = checkpoint_dir_path

        if not is_continue:
            self.__create_directories()
        self.__prefix = prefix

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

    def data_processor_state_file(self, preffix: str = None) -> str:
        """
        Get path of data processor state file
        """
        return os.path.join(self.optimizer_state_dir(), "{}_".format(preffix) if preffix is not None else "" + "dp_state.json")

    def logdir_path(self) -> str:
        """
        Get path of directory, there will be stored logs
        """
        return self.__logdir_path

    def __create_directories(self) -> None:
        if os.path.exists(self.__checkpoint_dir) and os.path.isdir(self.__checkpoint_dir):
            if os.listdir(self.__checkpoint_dir):
                raise self.FSMException("Checkpoint directory already exists [{}]".format(self.__checkpoint_dir))
        else:
            os.makedirs(self.__checkpoint_dir, exist_ok=True)

        if os.path.exists(self.__logdir_path) and os.path.isdir(self.__logdir_path):
            if os.listdir(self.__logdir_path):
                raise self.FSMException("Logs directory already exists [{}]".format(self.__logdir_path))
        else:
            os.makedirs(self.__logdir_path, exist_ok=True)
