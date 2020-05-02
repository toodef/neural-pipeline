import os
import shutil
import sys

from io import StringIO
from contextlib import contextmanager

import torch
from torch import Tensor
import numpy as np
import unittest

from piepline.utils import FileStructManager, CheckpointsManager, dict_recursive_bypass
from piepline.utils.fsm import FolderRegistrable
from tests.common import UseFileStructure

__all__ = ['UtilsTest', 'FileStructManagerTest', 'CheckpointsManagerTests']


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class UtilsTest(unittest.TestCase):
    def test_dict_recursive_bypass(self):
        d = {'data': np.array([1]), 'target': {'a': np.array([1]), 'b': np.array([1])}}
        d = dict_recursive_bypass(d, lambda v: torch.from_numpy(v))

        self.assertTrue(isinstance(d['data'], Tensor))
        self.assertTrue(isinstance(d['target']['a'], Tensor))
        self.assertTrue(isinstance(d['target']['b'], Tensor))


class FileStructManagerTest(UseFileStructure):
    class TestObj(FolderRegistrable):
        def __init__(self, m: 'FileStructManager', dir: str, name: str):
            super().__init__(m)
            self.dir = dir
            self.name = name

        def _get_gir(self) -> str:
            return self.dir

        def _get_name(self) -> str:
            return self.name

    def test_creation(self):
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.checkpoints_dir, ignore_errors=True)

        try:
            FileStructManager(base_dir=self.base_dir, is_continue=False)
        except FileStructManager.FSMException as err:
            self.fail("Raise error when base directory exists: [{}]".format(err))

        self.assertFalse(os.path.exists(self.base_dir))

        try:
            FileStructManager(base_dir=self.base_dir, is_continue=False)
        except FileStructManager.FSMException as err:
            self.fail("Raise error when base directory exists but empty: [{}]".format(err))

        os.makedirs(os.path.join(self.base_dir, 'new_dir'))
        try:
            FileStructManager(base_dir=self.base_dir, is_continue=False)
        except:
            self.fail("Error initialize when exists non-registered folders in base directory")

        shutil.rmtree(self.base_dir, ignore_errors=True)

    def test_object_registration(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        fsm_exist_ok = FileStructManager(base_dir=self.base_dir, is_continue=False, exists_ok=True)
        o = self.TestObj(fsm, 'test_dir', 'test_name')
        fsm.register_dir(o)

        expected_path = os.path.join(self.base_dir, 'test_dir')
        self.assertFalse(os.path.exists(expected_path))
        self.assertEqual(fsm.get_path(o), expected_path)

        with self.assertRaises(FileStructManager.FSMException):
            fsm.register_dir(self.TestObj(fsm, 'test_dir', 'another_name'))
        try:
            fsm.register_dir(self.TestObj(fsm, 'test_dir', 'another_name'), check_dir_registered=False)
            fsm_exist_ok.register_dir(self.TestObj(fsm, 'test_dir', 'another_name'))
            fsm_exist_ok.register_dir(self.TestObj(fsm, 'test_dir', 'another_name2'), check_dir_registered=False)
        except:
            self.fail("Folder registrable test fail when it's disabled")

        with self.assertRaises(FileStructManager.FSMException):
            fsm.register_dir(self.TestObj(fsm, 'another_dir', 'test_name'))
            fsm.register_dir(self.TestObj(fsm, 'another_dir', 'another_name'))
        with self.assertRaises(FileStructManager.FSMException):
            fsm_exist_ok.register_dir(self.TestObj(fsm, 'another_dir', 'test_name'))
            fsm_exist_ok.register_dir(self.TestObj(fsm, 'another_dir', 'another_name'))

        try:
            fsm.register_dir(self.TestObj(fsm, 'another_dir2', 'test_name'), check_name_registered=False)
            fsm_exist_ok.register_dir(self.TestObj(fsm, 'another_dir2', 'test_name'), check_name_registered=False)
        except:
            self.fail("Folder registrable test fail when it's disabled")

        os.makedirs(os.path.join(self.base_dir, 'dir_dir', 'dir'))
        with self.assertRaises(FileStructManager.FSMException):
            fsm.register_dir(self.TestObj(fsm, 'dir_dir', 'name_name'))

        try:
            fsm_exist_ok.register_dir(self.TestObj(fsm, 'dir_dir', 'name_name'))
        except:
            self.fail("Folder registrable test fail when exists_ok=True")


class CheckpointsManagerTests(UseFileStructure):
    def test_initialisation(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)

        try:
            cm = CheckpointsManager(fsm)
        except Exception as err:
            self.fail("Fail init CheckpointsManager; err: ['{}']".format(err))

        with self.assertRaises(FileStructManager.FSMException):
            CheckpointsManager(fsm)

        os.mkdir(os.path.join(fsm.get_path(cm), 'test_dir'))
        with self.assertRaises(FileStructManager.FSMException):
            CheckpointsManager(fsm)

    def test_pack(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        cm = CheckpointsManager(fsm)
        with self.assertRaises(CheckpointsManager.SMException):
            cm.pack()

        os.mkdir(cm.weights_file())
        os.mkdir(cm.optimizer_state_file())
        with self.assertRaises(CheckpointsManager.SMException):
            cm.pack()

        shutil.rmtree(cm.weights_file())
        shutil.rmtree(cm.optimizer_state_file())

        f = open(cm.weights_file(), 'w')
        f.close()
        f = open(cm.optimizer_state_file(), 'w')
        f.close()
        f = open(cm.trainer_file(), 'w')
        f.close()

        try:
            cm.pack()
        except Exception as err:
            self.fail('Exception on packing files: [{}]'.format(err))

        for f in [cm.weights_file(), cm.optimizer_state_file()]:
            if os.path.exists(f) and os.path.isfile(f):
                self.fail("File '{}' doesn't remove after pack".format(f))

        result = os.path.join(fsm.get_path(cm, check=False, create_if_non_exists=False), 'last_checkpoint.zip')
        self.assertTrue(os.path.exists(result) and os.path.isfile(result))

        f = open(cm.weights_file(), 'w')
        f.close()
        f = open(cm.optimizer_state_file(), 'w')
        f.close()
        f = open(cm.trainer_file(), 'w')
        f.close()

        try:
            cm.pack()
            result = os.path.join(fsm.get_path(cm, check=False, create_if_non_exists=False), 'last_checkpoint.zip.old')
            self.assertTrue(os.path.exists(result) and os.path.isfile(result))
        except Exception as err:
            self.fail('Fail to pack with existing previous state file')

    def test_unpack(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        cm = CheckpointsManager(fsm)

        f = open(cm.weights_file(), 'w')
        f.write('1')
        f.close()
        f = open(cm.optimizer_state_file(), 'w')
        f.write('2')
        f = open(cm.trainer_file(), 'w')
        f.write('3')
        f.close()

        cm.pack()

        try:
            cm.unpack()
        except Exception as err:
            self.fail('Exception on unpacking')

        for i, f in enumerate([cm.weights_file(), cm.optimizer_state_file(), cm.trainer_file()]):
            if not (os.path.exists(f) and os.path.isfile(f)):
                self.fail("File '{}' doesn't remove after pack".format(f))
            with open(f, 'r') as file:
                if file.read() != str(i + 1):
                    self.fail("File content corrupted")

    def test_clear_files(self):
        fsm = FileStructManager(base_dir=self.base_dir, is_continue=False)
        cm = CheckpointsManager(fsm)

        f = open(cm.weights_file(), 'w')
        f.close()
        f = open(cm.optimizer_state_file(), 'w')
        f.close()

        cm.clear_files()

        for f in [cm.weights_file(), cm.optimizer_state_file()]:
            if os.path.exists(f) and os.path.isfile(f):
                self.fail("File '{}' doesn't remove after pack".format(f))
