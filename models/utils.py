import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Normalization(data, mean, std):
    mean = mean.view(1,-1, 1, 1)
    std = std.view(1,-1, 1, 1)
    return (data.to(device)-mean.to(device))/std.to(device)

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.size()
        global_feature = self.global_pool(x)
        attention_weights = self.fc(global_feature).view(n, c, 1, 1)
        attended_feature = x * attention_weights
        return attended_feature


class GumbelSigmoid(nn.Module):
    def __init__(self, tau=1.0):
        super(GumbelSigmoid, self).__init__()

        self.tau = tau
        self.softmax = nn.Softmax(dim=1)
        self.p_value = 1e-8

    def forward(self, x, is_eval=False):
        r = 1 - x

        x = (x + self.p_value).log()
        r = (r + self.p_value).log()

        if not is_eval:
            x_N = torch.rand_like(x)
            r_N = torch.rand_like(r)
        else:
            x_N = 0.5 * torch.ones_like(x)
            r_N = 0.5 * torch.ones_like(r)

        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()

        x = x + x_N
        x = x / (self.tau + self.p_value)
        r = r + r_N
        r = r / (self.tau + self.p_value)

        x = torch.cat((x, r), dim=1)
        x = self.softmax(x)

        return x


class SRMFilter(nn.Module):
    def __init__(self, in_channel=64):
        super(SRMFilter, self).__init__()

        q = torch.tensor([4.0, 12.0, 2.0])

        filter1 = torch.tensor([[0, 0, 0, 0, 0],
                                [0, -1, 2, -1, 0],
                                [0, 2, -4, 2, 0],
                                [0, -1, 2, -1, 0],
                                [0, 0, 0, 0, 0]], dtype=torch.float32) / q[0]
        filter2 = torch.tensor([[-1, 2, -2, 2, -1],
                                [2, -6, 8, -6, 2],
                                [-2, 8, -12, 8, -2],
                                [2, -6, 8, -6, 2],
                                [-1, 2, -2, 2, -1]], dtype=torch.float32) / q[1]
        filter3 = torch.tensor([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 1, -2, 1, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]], dtype=torch.float32) / q[2]

        filters = torch.stack([filter1, filter2, filter3])
        filters = filters.repeat(in_channel, 1, 1).view(in_channel * 3, 1, 5, 5)
        # TODO: Warning groups=channel
        self.conv = nn.Conv2d(in_channel, in_channel * 3, kernel_size=5, padding=2, groups=in_channel, bias=False)
        self.conv.weight = nn.Parameter(filters, requires_grad=False)
        self.conv1 = nn.Conv2d(in_channel * 3, in_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channel * 3)

    def forward(self, x):
        feature = F.relu(self.bn1(self.conv(x)))
        feature = self.conv1(feature)
        mask = feature.reshape(feature.shape[0], 1, -1)
        mask = torch.nn.Sigmoid()(mask)
        mask = GumbelSigmoid(tau=0.1)(mask)
        mask = mask[:, 0].reshape(mask.shape[0], feature.shape[1], feature.shape[2], feature.shape[3])

        r_feat = feature * mask
        nr_feat = feature * (1 - mask)
        return r_feat, nr_feat, mask

class Separation(nn.Module):
    def __init__(self, in_channel=64):
        super(Separation, self).__init__()
        C = in_channel
        num_channel = 64
        self.sep_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )
    def forward(self,x):
        feature = self.sep_net(x)
        mask = feature.reshape(feature.shape[0], 1, -1)
        mask = torch.nn.Sigmoid()(mask)
        mask = GumbelSigmoid(tau=0.1)(mask)
        mask = mask[:, 0].reshape(mask.shape[0], feature.shape[1], feature.shape[2], feature.shape[3])

        HF_feat = feature * mask
        LF_feat = feature * (1 - mask)
        return HF_feat, LF_feat, mask


class Recalibration(nn.Module):
    def __init__(self, size, num_channel=64):
        super(Recalibration, self).__init__()
        self.rec_net = nn.Sequential(
            nn.Conv2d(size, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, size, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, feat, mask):
        rec_units = self.rec_net(feat)
        rec_units = rec_units * mask

        return rec_units
