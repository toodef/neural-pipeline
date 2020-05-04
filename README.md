# PiePline

Neural networks training pipeline based on PyTorch. Designed to standardize training process and accelerate experiments.

[![Build Status](https://travis-ci.org/PiePline/piepline.svg?branch=master)](https://travis-ci.org/PiePline/piepline)
[![Coverage Status](https://coveralls.io/repos/github/PiePline/piepline/badge.svg?branch=master)](https://coveralls.io/github/PiePline/piepline?branch=master)
[![Maintainability](https://api.codeclimate.com/v1/badges/7da18cb28e7e7dc13268/maintainability)](https://codeclimate.com/github/PiePline/piepline/maintainability)
[![Gitter chat](https://badges.gitter.im/piepline/gitter.png)](https://gitter.im/piepline/community)

* Core is about 2K lines, covered by tests, that you don't need to write again
* Flexible and customizable training process
* Checkpoints management and train process resuming (source and target device independent)
* Metrics processing and visualization by builtin ([tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard), [Matplotlib](https://matplotlib.org)) or custom monitors
* Training best practices (e.g. learning rate decaying and hard negative mining)
* Metrics logging and comparison (DVC compatible)

# Getting started:
### Documentation
[![Documentation Status](https://readthedocs.org/projects/piepline/badge/?version=stable)](https://piepline.readthedocs.io/en/stable/?badge=stable)
* [See the full documentation there](https://piepline.readthedocs.io/en/stable/)
* [Read getting started guide](https://piepline.readthedocs.io/en/stable/getting_started/index.html)

### See the examples
* MNIST classification - [notebook](https://github.com/toodef/piepline/blob/master/examples/notebooks/img_classification.ipynb), [file](https://github.com/toodef/piepline/blob/master/examples/files/img_classification.py), [Kaggle kernel](https://www.kaggle.com/toodef/cnn-training-with-less-code)
* Segmentation - [notebook](https://github.com/toodef/piepline/blob/master/examples/notebooks/img_segmentation.ipynb), [file](https://github.com/toodef/piepline/blob/master/examples/files/img_segmentation.py)
* Resume training process - [file](https://github.com/toodef/piepline/blob/master/examples/files/resume_train.py)

### PiePline short overview:
```python
import torch

from neural_pipeline.builtin.monitors.tensorboard import TensorboardMonitor
from neural_pipeline.monitoring import LogMonitor
from neural_pipeline import DataProducer, TrainConfig, TrainStage,\
    ValidationStage, Trainer, FileStructManager

from somethig import MyNet, MyDataset

fsm = FileStructManager(base_dir='data', is_continue=False)
model = MyNet().cuda()

train_dataset = DataProducer([MyDataset()], batch_size=4, num_workers=2)
validation_dataset = DataProducer([MyDataset()], batch_size=4, num_workers=2)

train_config = TrainConfig(model, [TrainStage(train_dataset),
                                   ValidationStage(validation_dataset)], torch.nn.NLLLoss(),
                           torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.5))

trainer = Trainer(train_config, fsm, torch.device('cuda:0')).set_epoch_num(50)
trainer.monitor_hub.add_monitor(TensorboardMonitor(fsm, is_continue=False))\
                   .add_monitor(LogMonitor(fsm))
trainer.train()
```
This example of training MyNet on MyDataset with vizualisation in Tensorflow and with metrics logging for further experiments comparison.

# Installation:
[![PyPI version](https://badge.fury.io/py/piepline.svg)](https://badge.fury.io/py/piepline)
[![PyPI Downloads/Month](https://pepy.tech/badge/piepline/month)](https://pepy.tech/project/piepline)
[![PyPI Downloads](https://pepy.tech/badge/piepline)](https://pepy.tech/project/piepline)

`pip install piepline`

##### For `builtin` module using install:
`pip install tensorboardX matplotlib`

##### Install latest version before it's published on PyPi
`pip install -U git+https://github.com/PiePline/piepline`
