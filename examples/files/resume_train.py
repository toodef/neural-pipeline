"""
This file contain example for resume training described in img_segmentation.py
"""

import torch
import numpy as np

from neural_pipeline import Trainer
from neural_pipeline.builtin.models.albunet import resnet18
from neural_pipeline.builtin.monitors.tensorboard import TensorboardMonitor
from neural_pipeline.monitoring import LogMonitor
from neural_pipeline.train_config import TrainConfig
from neural_pipeline.utils import FileStructManager

from examples.files.img_segmentation import train_stage, val_stage


def continue_training():
    ########################################################
    # Create needed parameters again
    ########################################################

    model = resnet18(classes_num=1, in_channels=3, pretrained=True)
    train_config = TrainConfig([train_stage, val_stage], torch.nn.BCEWithLogitsLoss(),
                               torch.optim.Adam(model.parameters(), lr=1e-4))

    ########################################################
    # If FileStructManager creates again - just 'set is_continue' parameter to True
    ########################################################
    file_struct_manager = FileStructManager(base_dir='data', is_continue=True)

    trainer = Trainer(model, train_config, file_struct_manager, torch.device('cuda:0')).set_epoch_num(10)

    tensorboard = TensorboardMonitor(file_struct_manager, is_continue=False, network_name='PortraitSegmentation')
    log = LogMonitor(file_struct_manager).write_final_metrics()
    trainer.monitor_hub.add_monitor(tensorboard).add_monitor(log)
    trainer.enable_best_states_saving(lambda: np.mean(train_stage.get_losses()))

    trainer.enable_lr_decaying(coeff=0.5, patience=10, target_val_clbk=lambda: np.mean(train_stage.get_losses()))
    trainer.add_on_epoch_end_callback(lambda: tensorboard.update_scalar('params/lr', trainer.data_processor().get_lr()))

    ########################################################
    # For set resume mode to Trainer just call 'resume' method
    ########################################################

    trainer.resume(from_best_checkpoint=False).train()
