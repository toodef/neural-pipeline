import os
from zipfile import ZipFile


class StateManager:
    def __init__(self, directory: str, preffix: str = None):
        self.__dir = directory
        self.__preffix = preffix

        self.__files = {}

    def unpack(self) -> None:
        self.__files['weights_file'] = os.path.join(self.__dir, "weights.pth")
        self.__files['state_file'] = os.path.join(self.__dir, "state.pth")
        result_file = self.__construct_result_file()

        with ZipFile(result_file, 'r') as zipfile:
            zipfile.extractall(self.__dir)

    def clear_files(self):
        def rm_file(file: str):
            if os.path.exists(file) and os.path.isfile(file):
                os.remove(file)

        rm_file(self.__files['weights_file'])
        rm_file(self.__files['state_file'])
        self.__files = {}

    def pack(self):
        def rm_file(file: str):
            if os.path.exists(file) and os.path.isfile(file):
                os.remove(file)

        def rename_file(file: str):
            target = file + ".old"
            rm_file(target)
            if os.path.exists(file) and os.path.isfile(file):
                os.rename(file, target)

        weights_file = os.path.join(self.__dir, "weights.pth")
        state_file = os.path.join(self.__dir, "state.pth")
        result_file = self.__construct_result_file()

        rm_file(weights_file)
        rm_file(state_file)

        rename_file(result_file)
        with ZipFile(result_file, 'w') as zipfile:
            zipfile.write(weights_file, os.path.basename(weights_file))
            zipfile.write(state_file, os.path.basename(state_file))

        rm_file(weights_file)
        rm_file(state_file)

    def get_files(self):
        return self.__files

    def __construct_result_file(self):
        return os.path.join(self.__dir, self.__preffix + "_" if self.__preffix is not None else "" + "state.zip")
