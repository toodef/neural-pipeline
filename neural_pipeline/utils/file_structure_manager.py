import os
from abc import ABCMeta, abstractmethod
from zipfile import ZipFile

__all__ = ['CheckpointsManager', 'FileStructManager', 'FolderRegistrable']


class FolderRegistrable(metaclass=ABCMeta):
    """
    Abstract class for implement classes, that use folders

    :param fsm: FileStructureManager class instance
    """

    @abstractmethod
    def __init__(self, fsm: 'FileStructManager'):
        pass

    @abstractmethod
    def get_gir(self) -> str:
        """
        Get directory path to register

        :return: path
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        Get name of registrable object

        :return: name
        """


class CheckpointsManager(FolderRegistrable):
    """
    Class that manage checkpoints for DataProcessor.

    All states pack to zip file. It contains few files: model weights, optimizer state, data processor state

    :param fsm: :class:'FileStructureManager' instance
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

    def __init__(self, fsm: 'FileStructManager', prefix: str = None):
        super().__init__(fsm)

        self._prefix = prefix if prefix is not None else 'last'
        fsm.register_dir(self)
        self._checkpoints_dir = fsm.get_path(self, create_if_non_exists=True, check=False)

        if (prefix is None) and (not (os.path.exists(self._checkpoints_dir) and os.path.isdir(self._checkpoints_dir))):
            raise self.SMException("Checkpoints dir doesn't exists: [{}]".format(self._checkpoints_dir))

        self._weights_file = os.path.join(self._checkpoints_dir, 'weights.pth')
        self._state_file = os.path.join(self._checkpoints_dir, 'state.pth')
        self._checkpoint_file = self._compile_path(self._checkpoints_dir, 'checkpoint.zip')

        if os.path.exists(self._weights_file) and os.path.exists(self._state_file) and \
                os.path.isfile(self._weights_file) and os.path.isfile(self._state_file):
            prev_prefix = self._prefix
            self._prefix = "prev_start"
            self.pack()
            self._prefix = prev_prefix

    def unpack(self) -> None:
        """
        Unpack state files
        """
        with ZipFile(self._checkpoint_file, 'r') as zipfile:
            zipfile.extractall(self._checkpoints_dir)

        self._check_files([self._weights_file, self._state_file])

    def clear_files(self) -> None:
        """
        Clear unpacked files
        """

        def rm_file(file: str):
            if os.path.exists(file) and os.path.isfile(file):
                os.remove(file)

        rm_file(self._weights_file)
        rm_file(self._state_file)

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

        self._check_files([self._weights_file, self._state_file])

        rename_file(self._checkpoint_file)
        with ZipFile(self._checkpoint_file, 'w') as zipfile:
            zipfile.write(self._weights_file, os.path.basename(self._weights_file))
            zipfile.write(self._state_file, os.path.basename(self._state_file))

        self.clear_files()

    def optimizer_state_file(self) -> str:
        return self._state_file

    def weights_file(self) -> str:
        return self._weights_file

    def _compile_path(self, directory: str, file: str) -> str:
        """
        Internal method for compile result file name

        :return: path to result file
        """
        return os.path.join(directory, (self._prefix + "_" if self._prefix is not None else "") + file)

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

    def get_gir(self) -> str:
        return os.path.join('checkpoints', self._prefix)

    def get_name(self) -> str:
        return 'CheckpointsManager' + self._prefix


class FileStructManager:
    """
    This class manage data directory. It's get path to config and provide info about folder and interface for work with it

    :param base_dir: path to directory with checkpoints
    :param is_continue: is FileStructManager used for continue training or predict
    """

    class FSMException(Exception):
        def __init__(self, message: str):
            self.__message = message

        def __str__(self):
            return self.__message

    class _Folder:
        def __init__(self, path: str, fsm: 'FileStructManager'):
            self._path = path
            self._fsm = fsm

            self._path_first_request = True

        def get_path_for_check(self) -> str:
            return self._path

        def _create_directories(self) -> None:
            if self._fsm._is_continue:
                return
            if not (os.path.exists(self._path) and os.path.isdir(self._path)):
                os.makedirs(self._path, exist_ok=True)

        def get_path(self, create_if_non_exists: bool = True) -> str:
            if create_if_non_exists and self._path_first_request:
                self._create_directories()
                self._path_first_request = False
            return self._path

        def check_path(self):
            if os.path.exists(self._path) and os.path.isdir(self._path):
                if os.listdir(self._path):
                    raise self._fsm.FSMException("Checkpoint directory already exists [{}]".format(self._path))

    def __init__(self, base_dir: str, is_continue: bool, exist_ok: bool = False):
        self._dirs = {}
        self._is_continue = is_continue
        self._base_dir = base_dir
        self._exist_ok = exist_ok

    def register_dir(self, obj: FolderRegistrable, check_name_registered: bool = True, check_dir_registered: bool = True) -> None:
        path = os.path.join(self._base_dir, obj.get_gir())

        if check_dir_registered:
            for n, f in self._dirs.items():
                if f.get_path_for_check() == path:
                    raise self.FSMException("Path {} already registered!".format(path))

        if check_name_registered:
            if obj.get_name() in self._dirs:
                raise self.FSMException("Object {} already registered!".format(obj.get_name()))

        self._dirs[obj.get_name()] = self._Folder(path, self)
        if not self._exist_ok and not self._is_continue:
            self._dirs[obj.get_name()].check_path()

    def get_path(self, obj: FolderRegistrable, create_if_non_exists: bool = False, check: bool = True) -> str:
        dir = self._dirs[obj.get_name()]
        if not self._exist_ok and check:
            dir.check_path()
        return dir.get_path(create_if_non_exists)
