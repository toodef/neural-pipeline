Neural networks training pipeline based on PyTorch 0.4.1 and designed to standardize and facilitate the training process.

[![Build Status](https://travis-ci.org/toodef/neural-pipeline.svg?branch=master)](https://travis-ci.org/toodef/neural-pipeline)
[![Coverage Status](https://coveralls.io/repos/github/toodef/neural-pipeline/badge.svg?branch=master)](https://coveralls.io/github/toodef/neural-pipeline?branch=master)
[![Maintainability](https://api.codeclimate.com/v1/badges/1feaafcc614adf27c30f/maintainability)](https://codeclimate.com/github/toodef/neural-pipeline/maintainability)

It's contains:
* Flexible and customizable training process
* Checkpoints management and train process resuming
* Metrics processing and visualisation by builtins ([tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard), [Matplotlib](https://matplotlib.org)) or custom monitors
* Training best practices (e.g. learning rate decaying and hard negative mining)
* Metrics logging and comparision

# Installation:
[![PyPI version](https://badge.fury.io/py/neural-pipeline.svg)](https://badge.fury.io/py/neural-pipeline)
[![PyPI Downloads/Month](https://pepy.tech/badge/neural-pipeline/month)](https://pepy.tech/project/neural-pipeline)
[![PyPI Downloads](https://pepy.tech/badge/neural-pipeline)](https://pepy.tech/project/neural-pipeline)

`pip install neural-pipeline`

##### For `builtin` module using install:
`pip install tensorboardX`

##### Install latest version before it's published on PyPi
`pip install -U git+https://github.com/toodef/neural-pipeline`

# Getting started:
### Documentation
[![Documentation Status](https://readthedocs.org/projects/neural-pipeline/badge/?version=master)](https://neural-pipeline.readthedocs.io/en/master/?badge=master)
[See the full documentation there](https://neural-pipeline.readthedocs.io/en/master/)

Data flow scheme:
[Data flow](docs/img/data_flow.svg)

### See the examples
* MNIST classification example - [notebook](https://github.com/toodef/neural-pipeline/blob/master/examples/notebooks/img_classification.ipynb), [file](https://github.com/toodef/neural-pipeline/blob/master/examples/src/img_classification.py)
* Segmentation example - [notebook](https://github.com/toodef/neural-pipeline/blob/master/examples/notebooks/img_segmentation.ipynb), [file](https://github.com/toodef/neural-pipeline/blob/master/examples/files/img_segmentation.py)



