import os
import shutil


class UseResources:
    logdir = 'logdir'
    checkpoints_dir = 'checkpoints_dir'

    def tearDown(self):
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir, ignore_errors=True)
        if os.path.exists(self.checkpoints_dir):
            shutil.rmtree(self.checkpoints_dir, ignore_errors=True)

