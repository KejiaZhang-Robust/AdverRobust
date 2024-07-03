import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *


class AlexNet_class(nn.Module):
    def __init__(self, num_classes=10, norm=False, mean=None, std=None):
        super(AlexNet_class, self).__init__()
        self._num_classes = num_classes
        # TODO: Implement AlexNet for cifar-10
        self.norm = norm
        self.mean = mean
        self.std = std
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.linear1 = nn.Linear(256 * 3 * 3, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, self._num_classes)
    
    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value
        self.fc = nn.Linear(512, self._num_classes).to(self.linear3.weight.device)

    def forward(self, x):
        x = x.to(device)
        if self.norm == True:
            x = Normalization(x, self.mean, self.std)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = F.relu(x, inplace=True)

        x = self.conv4(x)
        x = F.relu(x, inplace=True)

        x = self.conv5(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = torch.flatten(x, 1)

        x = nn.Dropout()(x)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)

        x = nn.Dropout()(x)
        x = self.linear2(x)
        x = F.relu(x, inplace=True)

        x = self.linear3(x)

        return x


def Alex_Net(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return AlexNet_class(num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

