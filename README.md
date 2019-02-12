# Neural Piepline

Neural networks training pipeline based on PyTorch. Designed to standardize training process and to increase coding preformance.

[![Build Status](https://travis-ci.org/toodef/neural-pipeline.svg?branch=master)](https://travis-ci.org/toodef/neural-pipeline)
[![Coverage Status](https://coveralls.io/repos/github/toodef/neural-pipeline/badge.svg?branch=master)](https://coveralls.io/github/toodef/neural-pipeline?branch=master)
[![Maintainability](https://api.codeclimate.com/v1/badges/1feaafcc614adf27c30f/maintainability)](https://codeclimate.com/github/toodef/neural-pipeline/maintainability)

* Core is about 2K lines, covered by tests, that you doesn't need to write again
* Flexible and customizable training process
* Checkpoints management and train process resuming (source and target device independent)
* Metrics processing and visualization by builtin ([tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard), [Matplotlib](https://matplotlib.org)) or custom monitors
* Training best practices (e.g. learning rate decaying and hard negative mining)
* Metrics logging and comparison (DVC compatible)

# Train MNIST example:
This code run MNIST image classification with Tensorboard monitoring. Code based on PyTorch [example](https://github.com/pytorch/examples/blob/master/mnist/main.py).

See full example [there](https://github.com/toodef/neural-pipeline/blob/master/examples/files/img_classification.py).
```python
from neural_pipeline.builtin.monitors.tensorboard import TensorboardMonitor
from neural_pipeline import DataProducer, AbstractDataset, TrainConfig, TrainStage,\
    ValidationStage, Trainer, FileStructManager

import torch
from torch import nn
from torchvision import datasets, transforms

class Net(nn.Module):
    # Network implementation

class MNISTDataset(AbstractDataset):
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def __init__(self, data_dir: str, is_train: bool):
        self.dataset = datasets.MNIST(data_dir, train=is_train, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data, target = self.dataset[item]
        return {'data': self.transforms(data), 'target': target}

fsm = FileStructManager(base_dir='data', is_continue=False)
model = Net()

train_dataset = DataProducer([MNISTDataset('data/dataset', True)], batch_size=4, num_workers=2)
validation_dataset = DataProducer([MNISTDataset('data/dataset', False)], batch_size=4, num_workers=2)

train_config = TrainConfig([TrainStage(train_dataset), ValidationStage(validation_dataset)], torch.nn.NLLLoss(),
                           torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.5))

trainer = Trainer(model, train_config, fsm, torch.device('cuda:0')).set_epoch_num(50)
trainer.monitor_hub.add_monitor(TensorboardMonitor(fsm, is_continue=False))
trainer.train()
```

# Installation:
[![PyPI version](https://badge.fury.io/py/neural-pipeline.svg)](https://badge.fury.io/py/neural-pipeline)
[![PyPI Downloads/Month](https://pepy.tech/badge/neural-pipeline/month)](https://pepy.tech/project/neural-pipeline)
[![PyPI Downloads](https://pepy.tech/badge/neural-pipeline)](https://pepy.tech/project/neural-pipeline)

`pip install neural-pipeline`

##### For `builtin` module using install:
`pip install tensorboardX matplotlib`

##### Install latest version before it's published on PyPi
`pip install -U git+https://github.com/toodef/neural-pipeline`

# Getting started:
### Documentation
[![Documentation Status](https://readthedocs.org/projects/neural-pipeline/badge/?version=master)](https://neural-pipeline.readthedocs.io/en/master/?badge=master)
[See the full documentation there](https://neural-pipeline.readthedocs.io/en/master/)

Data flow scheme:
![Data flow](https://github.com/toodef/neural-pipeline/blob/master/docs/img/data_flow.svg)

### See the examples
* MNIST classification - [notebook](https://github.com/toodef/neural-pipeline/blob/master/examples/notebooks/img_classification.ipynb), [file](https://github.com/toodef/neural-pipeline/blob/master/examples/files/img_classification.py)
* Segmentation - [notebook](https://github.com/toodef/neural-pipeline/blob/master/examples/notebooks/img_segmentation.ipynb), [file](https://github.com/toodef/neural-pipeline/blob/master/examples/files/img_segmentation.py)
* Resume training process - [file](https://github.com/toodef/neural-pipeline/blob/master/examples/files/resume_train.py)



