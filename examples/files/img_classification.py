from neural_pipeline.builtin.monitors.tensorboard import TensorboardMonitor
from neural_pipeline import DataProducer, AbstractDataset, TrainConfig, TrainStage,\
    ValidationStage, Trainer, FileStructManager

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms


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


class MNISTDataset(AbstractDataset):
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def __init__(self, data_dir: str, is_train: bool):
        self.dataset = datasets.MNIST(data_dir, train=is_train, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data, target = self.dataset[item]
        return {'data': self.transforms(data), 'target': target}


if __name__ == '__main__':
    fsm = FileStructManager(base_dir='data', is_continue=False)
    model = Net()

    train_dataset = DataProducer([MNISTDataset('data/dataset', True)], batch_size=4, num_workers=2)
    validation_dataset = DataProducer([MNISTDataset('data/dataset', False)], batch_size=4, num_workers=2)

    train_config = TrainConfig(model, [TrainStage(train_dataset), ValidationStage(validation_dataset)], torch.nn.NLLLoss(),
                               torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.5))

    trainer = Trainer(train_config, fsm, torch.device('cuda:0')).set_epoch_num(5)
    trainer.monitor_hub.add_monitor(TensorboardMonitor(fsm, is_continue=False))
    trainer.train()
