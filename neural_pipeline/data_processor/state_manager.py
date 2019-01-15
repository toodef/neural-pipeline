import os
from zipfile import ZipFile

from neural_pipeline.utils.file_structure_manager import FileStructManager

__all__ = ['StateManager']


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

    def __init__(self, file_struct_manager: FileStructManager, prefix: str = None):
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
