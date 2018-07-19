import os
from zipfile import ZipFile

from tonet.tonet.utils.file_structure_manager import FileStructManager


class StateManager:
    def __init__(self, file_struct_manager: FileStructManager, preffix: str = None):
        self.__preffix = preffix
        self.__file_struct_manager = file_struct_manager

        self.__files = {}

    def unpack(self) -> None:
        self.__files['weights_file'] = os.path.join(self.__file_struct_manager.weights_dir(), "weights.pth")
        self.__files['state_file'] = os.path.join(self.__file_struct_manager.weights_dir(), "state.pth")
        result_file = self.__construct_result_file()

        with ZipFile(result_file, 'r') as zipfile:
            zipfile.extractall(self.__file_struct_manager.weights_dir())

    def clear_files(self) -> None:
        def rm_file(file: str):
            if os.path.exists(file) and os.path.isfile(file):
                os.remove(file)

        rm_file(self.__files['weights_file'])
        rm_file(self.__files['state_file'])
        self.__files = {}

    def pack(self) -> None:
        def rm_file(file: str):
            if os.path.exists(file) and os.path.isfile(file):
                os.remove(file)

        def rename_file(file: str):
            target = file + ".old"
            rm_file(target)
            if os.path.exists(file) and os.path.isfile(file):
                os.rename(file, target)

        weights_file = self.__file_struct_manager.weights_file()
        state_file = self.__file_struct_manager.weights_dir()
        result_file = self.__construct_result_file()

        rm_file(weights_file)
        rm_file(state_file)

        rename_file(result_file)
        with ZipFile(result_file, 'w') as zipfile:
            zipfile.write(weights_file, os.path.basename(weights_file))
            zipfile.write(state_file, os.path.basename(state_file))

        rm_file(weights_file)
        rm_file(state_file)

    def get_files(self) -> {'weights_file', 'state_file'}:
        return self.__files

    def __construct_result_file(self):
        data_dir = self.__file_struct_manager.weights_dir()
        return os.path.join(data_dir, self.__preffix + "_" if self.__preffix is not None else "" + "state.zip")
