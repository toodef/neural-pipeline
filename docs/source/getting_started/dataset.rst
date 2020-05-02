Implement dataset class
=======================

In PiePline dataset is iterable class. This means, that class need contain ``__getitem__`` and ``__len__`` methods.

For every i-th output, dataset need produce Python ``dict`` with keys 'data' and 'target'.

Let's create MNIST dataset, based on builtin PyTorch dataset:

.. code:: python

    from torchvision import datasets, transforms

    class MNISTDataset(AbstractDataset):
        # define transforms
        transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        def __init__(self, data_dir: str, is_train: bool):
            # instantiate PyTorch dataset
            self.dataset = datasets.MNIST(data_dir, train=is_train, download=True)

        # define method, that output dataset length
        def __len__(self):
            return len(self.dataset)

        # define method, that return single data by index
        def __getitem__(self, item):
            data, target = self.dataset[item]
            return {'data': self.transforms(data), 'target': target}

For work with this dataset we need wrap it by ``DataProducer``:

.. code:: python

    from piepline import DataProducer

    # create train and validation datasets objects
    train_dataset = DataProducer([MNISTDataset('data/dataset', True)], batch_size=4, num_workers=2)
    validation_dataset = DataProducer([MNISTDataset('data/dataset', False)], batch_size=4, num_workers=2)