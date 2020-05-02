Create Trainer
==================

First of all we need specify model, that will be trained:

.. code:: python

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)


Now we need build our training process. It's done by implements ``Trainer`` class:

.. code:: python

    from piepline import FileStructManager, Trainer

    # define file structure for experiment
    fsm = FileStructManager(base_dir='data', is_continue=False)

    # create trainer
    trainer = Trainer(model, train_config, fsm, torch.device('cuda:0'))

    # specify training epochs number
    trainer.set_epoch_num(50)

Last parameter or ``Trainer`` constructor - target device, that will be used for training.