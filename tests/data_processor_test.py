import os
import unittest
import shutil

from neural_pipeline.data_processor.state_manager import StateManager
from neural_pipeline.utils.file_structure_manager import FileStructManager

__all__ = ['DataProcessorTest']


class DataProcessorTest(unittest.TestCase):
    def test_state_manager(self):
        checkpoints_dir = 'checkpoints'
        logdir = 'logs'
        if os.path.exists(checkpoints_dir):
            shutil.rmtree(checkpoints_dir, ignore_errors=True)
        if os.path.exists(logdir):
            shutil.rmtree(logdir, ignore_errors=True)
        self.assertRaises(FileStructManager(checkpoint_dir_path=checkpoints_dir, logdir_path=logdir, prefix=None))

        os.mkdir(checkpoints_dir)
        try:
            fsm = FileStructManager(checkpoint_dir_path=checkpoints_dir, logdir_path=logdir, prefix=None)
        except:
            self.fail('FileStructManager raises exception for existing directory')
        state_manager = StateManager()
