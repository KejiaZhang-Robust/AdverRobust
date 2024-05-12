import torch
import torch.nn as nn

from .utils import *

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=100, norm = False, mean = None, std = None):
        super().__init__()
        self.features = features
        self.norm = norm
        self.mean = mean
        self.std = std
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        if self.norm == True:
            x = Normalization(x, self.mean, self.std)
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

def vgg13_bn(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return VGG(make_layers(cfg['B'], batch_norm=True), num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

def vgg16_bn(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)

def vgg19_bn(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)


