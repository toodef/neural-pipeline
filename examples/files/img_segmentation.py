"""
The images human portrait segmentation example.

Images dataset was taken from [PicsArt](https://picsart.com/) AI Hackathon.

Dataset may be downloaded [there](https://s3.eu-central-1.amazonaws.com/datasouls/public/picsart_hack_online_data.zip)

For this example need to install this dependencies:
`pip install sklearn, albumentations, opencv-python`
"""

import torch

import cv2
import os
import numpy as np

from sklearn.model_selection import train_test_split

from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, RandomGamma, \
    RandomBrightnessContrast, RGBShift, Resize, RandomCrop, OneOf

from neural_pipeline import Trainer
from neural_pipeline.builtin.models.albunet import resnet18
from neural_pipeline.data_producer import AbstractDataset, DataProducer
from neural_pipeline.monitoring import LogMonitor
from neural_pipeline.train_config import AbstractMetric, MetricsProcessor, MetricsGroup, TrainStage, ValidationStage, TrainConfig
from neural_pipeline.utils.fsm import FileStructManager
from neural_pipeline.builtin.monitors.tensorboard import TensorboardMonitor

###################################
# Define dataset and augmentations
# The dataset used in this example is from PicsArt hackathon (https://picsart.ai/en/contest)
###################################

datasets_dir = 'data/dataset'
base_dir = os.path.join(datasets_dir, 'picsart_hack_online_data')

preprocess = OneOf([RandomCrop(height=224, width=224), Resize(width=224, height=224)], p=1)
transforms = Compose([HorizontalFlip(), VerticalFlip(), RandomRotate90()], p=0.5)
aug = Compose([RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4), RandomGamma(), RGBShift(), transforms])


def augmentate(item: {}):
    res = preprocess(image=item['data'], mask=item['target'])
    res = aug(image=res['image'], mask=res['mask'])
    return {'data': res['image'], 'target': res['mask']}


def augmentate_and_to_pytorch(item: {}):
    res = augmentate(item)
    return {'data': torch.from_numpy(np.moveaxis(res['data'].astype(np.float32) / 255., -1, 0)),
            'target': torch.from_numpy(np.expand_dims(res['target'].astype(np.float32) / 255, axis=0))}


class PicsartDataset(AbstractDataset):
    def __init__(self, images_pathes: [], aug: callable):
        images_dir = os.path.join(base_dir, 'train')
        masks_dir = os.path.join(base_dir, 'train_mask')
        images_pathes = sorted(images_pathes, key=lambda p: int(os.path.splitext(p)[0]))
        self.__image_pathes = []
        self.__aug = aug
        for p in images_pathes:
            name = os.path.splitext(p)[0]
            mask_img = os.path.join(masks_dir, name + '.png')
            if os.path.exists(mask_img):
                path = {'data': os.path.join(images_dir, p), 'target': mask_img}
                self.__image_pathes.append(path)

    def __len__(self):
        return len(self.__image_pathes)

    def __getitem__(self, item):
        img = cv2.imread(self.__image_pathes[item]['data'])
        return self.__aug({'data': img,
                           'target': cv2.imread(self.__image_pathes[item]['target'], cv2.IMREAD_UNCHANGED)})


img_dir = os.path.join(base_dir, 'train')
img_pathes = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1] == ".jpg"]
train_pathes, val_pathes = train_test_split(img_pathes, shuffle=True, test_size=0.2)

train_dataset, val_dataset = PicsartDataset(train_pathes, augmentate_and_to_pytorch), PicsartDataset(val_pathes,
                                                                                                     augmentate_and_to_pytorch)

###################################
# define metrics
###################################

eps = 1e-6


def dice(preds: torch.Tensor, trues: torch.Tensor) -> np.ndarray:
    preds_inner = preds.data.cpu().numpy().copy()
    trues_inner = trues.data.cpu().numpy().copy()

    preds_inner = np.reshape(preds_inner, (preds_inner.shape[0], preds_inner.size // preds_inner.shape[0]))
    trues_inner = np.reshape(trues_inner, (trues_inner.shape[0], trues_inner.size // trues_inner.shape[0]))

    intersection = (preds_inner * trues_inner).sum(1)
    scores = (2. * intersection + eps) / (preds_inner.sum(1) + trues_inner.sum(1) + eps)

    return scores


def jaccard(preds: torch.Tensor, trues: torch.Tensor):
    preds_inner = preds.cpu().data.numpy().copy()
    trues_inner = trues.cpu().data.numpy().copy()

    preds_inner = np.reshape(preds_inner, (preds_inner.shape[0], preds_inner.size // preds_inner.shape[0]))
    trues_inner = np.reshape(trues_inner, (trues_inner.shape[0], trues_inner.size // trues_inner.shape[0]))
    intersection = (preds_inner * trues_inner).sum(1)
    scores = (intersection + eps) / ((preds_inner + trues_inner).sum(1) - intersection + eps)

    return scores


class DiceMetric(AbstractMetric):
    def __init__(self):
        super().__init__('dice')

    def calc(self, output: torch.Tensor, target: torch.Tensor) -> np.ndarray or float:
        return dice(output, target)


class JaccardMetric(AbstractMetric):
    def __init__(self):
        super().__init__('jaccard')

    def calc(self, output: torch.Tensor, target: torch.Tensor) -> np.ndarray or float:
        return jaccard(output, target)


class SegmentationMetricsProcessor(MetricsProcessor):
    def __init__(self, stage_name: str):
        super().__init__()
        self.add_metrics_group(MetricsGroup(stage_name).add(JaccardMetric()).add(DiceMetric()))


###################################
# define train config and train model
###################################

train_data_producer = DataProducer([train_dataset], batch_size=2, num_workers=3)
val_data_producer = DataProducer([val_dataset], batch_size=2, num_workers=3)

train_stage = TrainStage(train_data_producer, SegmentationMetricsProcessor('train')).enable_hard_negative_mining(0.1)
val_metrics_processor = SegmentationMetricsProcessor('validation')
val_stage = ValidationStage(val_data_producer, val_metrics_processor)


def train():
    model = resnet18(classes_num=1, in_channels=3, pretrained=True)
    train_config = TrainConfig(model, [train_stage, val_stage], torch.nn.BCEWithLogitsLoss(),
                               torch.optim.Adam(model.parameters(), lr=1e-4))

    file_struct_manager = FileStructManager(base_dir='data', is_continue=False)

    trainer = Trainer(train_config, file_struct_manager, torch.device('cuda:0')).set_epoch_num(2)

    tensorboard = TensorboardMonitor(file_struct_manager, is_continue=False, network_name='PortraitSegmentation')
    log = LogMonitor(file_struct_manager).write_final_metrics()
    trainer.monitor_hub.add_monitor(tensorboard).add_monitor(log)
    trainer.enable_best_states_saving(lambda: np.mean(train_stage.get_losses()))

    trainer.enable_lr_decaying(coeff=0.5, patience=10, target_val_clbk=lambda: np.mean(train_stage.get_losses()))
    trainer.add_on_epoch_end_callback(lambda: tensorboard.update_scalar('params/lr', trainer.data_processor().get_lr()))
    trainer.train()


if __name__ == "__main__":
    train()
