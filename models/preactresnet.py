import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
class PreActBasic(nn.Module):

    expansion = 1
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * PreActBasic.expansion, kernel_size=3, padding=1)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBasic.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBasic.expansion, 1, stride=stride)

    def forward(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return res + shortcut


class PreActBottleNeck(nn.Module):

    expansion = 4
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, stride=stride),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * PreActBottleNeck.expansion, 1)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * PreActBottleNeck.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBottleNeck.expansion, 1, stride=stride)

    def forward(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return res + shortcut

class PreActResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, norm = False, mean = None, std = None):
        super().__init__()
        self.input_channels = 64
        self._num_classes = num_classes
        self.norm = norm
        self.mean = mean
        self.std = std

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_layers(block, num_block[0], 64,  1)
        self.stage2 = self._make_layers(block, num_block[1], 128, 2)
        self.stage3 = self._make_layers(block, num_block[2], 256, 2)
        self.stage4 = self._make_layers(block, num_block[3], 512, 2)

        self.linear = nn.Linear(self.input_channels, self._num_classes)

    def _make_layers(self, block, block_num, out_channels, stride):
        layers = []

        layers.append(block(self.input_channels, out_channels, stride))
        self.input_channels = out_channels * block.expansion

        while block_num - 1:
            layers.append(block(self.input_channels, out_channels, 1))
            self.input_channels = out_channels * block.expansion
            block_num -= 1

        return nn.Sequential(*layers)
    
    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value
        self.linear = nn.Linear(self.input_channels, self._num_classes).to(self.linear.weight.device)

    def forward(self, x):
        if self.norm == True:
            x = Normalization(x, self.mean, self.std)
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


def preactresnet18(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return PreActResNet(PreActBasic, [2, 2, 2, 2], num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

def preactresnet34(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return PreActResNet(PreActBasic, [3, 4, 6, 3], num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

def preactresnet50(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return PreActResNet(PreActBottleNeck, [3, 4, 6, 3], num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

def preactresnet101(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return PreActResNet(PreActBottleNeck, [3, 4, 23, 3], num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

def preactresnet152(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return PreActResNet(PreActBottleNeck, [3, 8, 36, 3], num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

