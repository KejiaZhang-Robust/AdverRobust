import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.num_classes = num_classes

    def forward(self, x):
        x = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        x = nn.Conv2d(64, 192, kernel_size=5, padding=2)(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        x = nn.Conv2d(192, 384, kernel_size=3, padding=1)(x)
        x = F.relu(x, inplace=True)
        
        x = nn.Conv2d(384, 256, kernel_size=3, padding=1)(x)
        x = F.relu(x, inplace=True)
        
        x = nn.Conv2d(256, 256, kernel_size=3, padding=1)(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = nn.Dropout()(x)
        x = nn.Linear(256 * 6 * 6, 4096)(x)
        x = F.relu(x, inplace=True)
        
        x = nn.Dropout()(x)
        x = nn.Linear(4096, 4096)(x)
        x = F.relu(x, inplace=True)
        
        x = nn.Linear(4096, self.num_classes)(x)
        
        return x
    
def AlexNet(Num_class=10, Norm=True, norm_mean=None, norm_std=None):
    return AlexNet(num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)