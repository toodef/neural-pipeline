import os
import shutil
import sys

from io import StringIO
from contextlib import contextmanager

import torch
from torch import Tensor
import numpy as np
import unittest

from neural_pipeline.utils.file_structure_manager import FileStructManager
from neural_pipeline.utils.utils import dict_recursive_bypass

__all__ = ['UtilsTest']


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


if __name__ == '__main__':
    unittest.main()
