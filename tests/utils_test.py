import os
import shutil
import sys

from io import StringIO
from contextlib import contextmanager

import torch
from torch import Tensor
import numpy as np
import unittest

from neural_pipeline.utils import FileStructManager, StateManager, dict_recursive_bypass

__all__ = ['UtilsTest', 'StateManagerTests']


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
    logdir = 'logs'
    checkpoints_dir = 'checkpoints'

    def test_dict_recursive_bypass(self):
        d = {'data': np.array([1]), 'target': {'a': np.array([1]), 'b': np.array([1])}}
        d = dict_recursive_bypass(d, lambda v: torch.from_numpy(v))

        self.assertTrue(isinstance(d['data'], Tensor))
        self.assertTrue(isinstance(d['target']['a'], Tensor))
        self.assertTrue(isinstance(d['target']['b'], Tensor))

    def test_file_struct_manager(self):
        if os.path.exists(self.checkpoints_dir):
            shutil.rmtree(self.checkpoints_dir, ignore_errors=True)
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir, ignore_errors=True)

        try:
            FileStructManager(checkpoint_dir_path=self.checkpoints_dir, logdir_path=self.logdir, prefix=None)
        except FileStructManager.FSMException as err:
            self.fail("Raise error when checkpoints and logs directories exists: [{}]".format(err))

        self.assertTrue(os.path.exists(self.checkpoints_dir))
        self.assertTrue(os.path.exists(self.logdir))

        shutil.rmtree(self.logdir, ignore_errors=True)
        try:
            FileStructManager(checkpoint_dir_path=self.checkpoints_dir, logdir_path=self.logdir, prefix=None)
        except FileStructManager.FSMException as err:
            self.fail("Raise error when checkpoints directory exists but empty: [{}]".format(err))

        shutil.rmtree(self.checkpoints_dir, ignore_errors=True)
        try:
            FileStructManager(checkpoint_dir_path=self.checkpoints_dir, logdir_path=self.logdir, prefix=None)
        except FileStructManager.FSMException as err:
            self.fail("Raise error when logs directory exists but empty: [{}]".format(err))

        os.mkdir(os.path.join(self.checkpoints_dir, 'new_dir'))
        with self.assertRaises(FileStructManager.FSMException):
            FileStructManager(checkpoint_dir_path=self.checkpoints_dir, logdir_path=self.logdir, prefix=None)
        shutil.rmtree(self.checkpoints_dir, ignore_errors=True)
        os.mkdir(os.path.join(self.logdir, 'new_dir'))
        with self.assertRaises(FileStructManager.FSMException):
            FileStructManager(checkpoint_dir_path=self.checkpoints_dir, logdir_path=self.logdir, prefix=None)

        shutil.rmtree(self.logdir, ignore_errors=True)
        shutil.rmtree(self.checkpoints_dir, ignore_errors=True)
        sfm = FileStructManager(checkpoint_dir_path=self.checkpoints_dir, logdir_path=self.logdir, prefix=None)

        self.assertEqual(sfm.checkpoint_dir(), self.checkpoints_dir)
        self.assertEqual(sfm.weights_file(), os.path.join(self.checkpoints_dir, 'weights.pth'))
        self.assertEqual(sfm.optimizer_state_dir(), self.checkpoints_dir)
        self.assertEqual(sfm.optimizer_state_file(), os.path.join(self.checkpoints_dir, 'state.pth'))
        self.assertEqual(sfm.data_processor_state_file(), os.path.join(self.checkpoints_dir, 'dp_state.json'))
        self.assertEqual(sfm.logdir_path(), self.logdir)

    def tearDown(self):
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir, ignore_errors=True)
        if os.path.exists(self.checkpoints_dir):
            shutil.rmtree(self.checkpoints_dir, ignore_errors=True)


class StateManagerTests(unittest.TestCase):
    logdir = 'logs'
    checkpoints_dir = 'checkpoints'

    def test_initialisation(self):
        fsm = FileStructManager(checkpoint_dir_path=self.checkpoints_dir, logdir_path=self.logdir, prefix=None)

        shutil.rmtree(self.checkpoints_dir)
        with self.assertRaises(StateManager.SMException):
            StateManager(fsm)

        os.mkdir(self.checkpoints_dir)
        try:
            sm = StateManager(fsm)
        except Exception as err:
            self.fail("Fail init StateManager; err: ['{}']".format(err))

    def test_pack(self):
        fsm = FileStructManager(checkpoint_dir_path=self.checkpoints_dir, logdir_path=self.logdir, prefix=None)
        sm = StateManager(fsm)
        with self.assertRaises(StateManager.SMException):
            sm.pack()

        os.mkdir(fsm.weights_file())
        os.mkdir(fsm.optimizer_state_file())
        os.mkdir(fsm.data_processor_state_file())
        with self.assertRaises(StateManager.SMException):
            sm.pack()

        shutil.rmtree(fsm.weights_file())
        shutil.rmtree(fsm.optimizer_state_file())
        shutil.rmtree(fsm.data_processor_state_file())

        f = open(fsm.weights_file(), 'w')
        f.write('1')
        f.close()
        f = open(fsm.optimizer_state_file(), 'w')
        f.write('1')
        f.close()
        f = open(fsm.data_processor_state_file(), 'w')
        f.write('1')
        f.close()

        try:
            sm.pack()
        except Exception as err:
            self.fail('Exception on packing files: [{}]'.format(err))

        for f in [fsm.weights_file(), fsm.optimizer_state_file(), fsm.data_processor_state_file()]:
            if os.path.exists(f) and os.path.isfile(f):
                self.fail("File '{}' doesn't remove after pack".format(f))

        result = os.path.join(fsm.checkpoint_dir(), 'state.zip')
        self.assertTrue(os.path.exists(result) and os.path.isfile(result))

        f = open(fsm.weights_file(), 'w')
        f.write('1')
        f.close()
        f = open(fsm.optimizer_state_file(), 'w')
        f.write('1')
        f.close()
        f = open(fsm.data_processor_state_file(), 'w')
        f.write('1')
        f.close()

        try:
            sm.pack()
            result = os.path.join(fsm.checkpoint_dir(), 'state.zip.old')
            self.assertTrue(os.path.exists(result) and os.path.isfile(result))
        except Exception as err:
            self.fail('Fail to pack with existing previous state file')

    def test_unpack(self):
        fsm = FileStructManager(checkpoint_dir_path=self.checkpoints_dir, logdir_path=self.logdir, prefix=None)
        sm = StateManager(fsm)

        f = open(fsm.weights_file(), 'w')
        f.write('1')
        f.close()
        f = open(fsm.optimizer_state_file(), 'w')
        f.write('2')
        f.close()
        f = open(fsm.data_processor_state_file(), 'w')
        f.write('3')
        f.close()

        sm.pack()

        try:
            sm.unpack()
        except Exception as err:
            self.fail('Exception on unpacking')

        for i, f in enumerate([fsm.weights_file(), fsm.optimizer_state_file(), fsm.data_processor_state_file()]):
            if not (os.path.exists(f) and os.path.isfile(f)):
                self.fail("File '{}' doesn't remove after pack".format(f))
            with open(f, 'r') as file:
                if file.read() != str(i + 1):
                    self.fail("File content corrupted")

    def test_clear_files(self):
        fsm = FileStructManager(checkpoint_dir_path=self.checkpoints_dir, logdir_path=self.logdir, prefix=None)
        sm = StateManager(fsm)

        f = open(fsm.weights_file(), 'w')
        f.close()
        f = open(fsm.optimizer_state_file(), 'w')
        f.close()
        f = open(fsm.data_processor_state_file(), 'w')
        f.close()

        sm.clear_files()

        for f in [fsm.weights_file(), fsm.optimizer_state_file(), fsm.data_processor_state_file()]:
            if os.path.exists(f) and os.path.isfile(f):
                self.fail("File '{}' doesn't remove after pack".format(f))

    def tearDown(self):
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir, ignore_errors=True)
        if os.path.exists(self.checkpoints_dir):
            shutil.rmtree(self.checkpoints_dir, ignore_errors=True)
