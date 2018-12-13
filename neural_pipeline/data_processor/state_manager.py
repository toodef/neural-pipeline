import os
from zipfile import ZipFile

from neural_pipeline.utils.file_structure_manager import FileStructManager

__all__ = ['StateManager']


class StateManager:
    """
    Class that manage states for data processor files.
    All states pack to zip file. It contains few files: model weights, optimizer state, data processor state
    """

    def __init__(self, file_struct_manager: FileStructManager, prefix: str = None):
        """
        :param file_struct_manager: file structure manager
        :param prefix: prefix of saved and loaded files
        """
        self.__file_struct_manager = file_struct_manager

        weights_file = self.__file_struct_manager.weights_file()
        state_file = self.__file_struct_manager.optimizer_state_file()

        if os.path.exists(weights_file) and os.path.exists(state_file) and os.path.isfile(weights_file) and os.path.isfile(state_file):
            self.__preffix = "prev_start"
            self.pack()
            self.__preffix = None

        self.__preffix = prefix

    def unpack(self) -> None:
        """
        Unpack state file
        """
        result_file = self.__construct_result_file()

        with ZipFile(result_file, 'r') as zipfile:
            zipfile.extractall(self.__file_struct_manager.checkpoint_dir())

    def clear_files(self) -> None:
        """
        Clear unpacked files
        """
        def rm_file(file: str):
            if os.path.exists(file) and os.path.isfile(file):
                os.remove(file)

        rm_file(self.__file_struct_manager.weights_file())
        rm_file(self.__file_struct_manager.optimizer_state_file())
        rm_file(self.__file_struct_manager.data_processor_state_file())

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

        weights_file = self.__file_struct_manager.weights_file()
        state_file = self.__file_struct_manager.optimizer_state_file()
        dp_state_file = self.__file_struct_manager.data_processor_state_file()
        result_file = self.__construct_result_file()

        rename_file(result_file)
        with ZipFile(result_file, 'w') as zipfile:
            zipfile.write(weights_file, os.path.basename(weights_file))
            zipfile.write(state_file, os.path.basename(state_file))
            zipfile.write(dp_state_file, os.path.basename(dp_state_file))

        rm_file(weights_file)
        rm_file(state_file)
        rm_file(dp_state_file)

    def get_files(self) -> {}:
        """
        Get files pathes
        """
        return {'weights_file': self.__file_struct_manager.weights_file(),
                'state_file': self.__file_struct_manager.optimizer_state_file()}

    def __construct_result_file(self) -> str:
        """
        Construct result file name
        :return: path to result file
        """
        data_dir = self.__file_struct_manager.checkpoint_dir()
        return os.path.join(data_dir, (self.__preffix + "_" if self.__preffix is not None else "") + "state.zip")
