import os
import shutil
import unittest


__all__ = ['UseFileStructure', 'data_remove']


class UseFileStructure(unittest.TestCase):
    base_dir = 'data'
    monitors_dir = 'monitors'
    checkpoints_dir = 'checkpoints_dir'

    def tearDown(self):
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir, ignore_errors=True)


def data_remove(func: callable) -> callable:
    def res(*args, **kwargs):
        ret = func(*args, **kwargs)
        UseFileStructure().tearDown()
        return ret

    return res
