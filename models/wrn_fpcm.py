import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Normalization(data, mean, std):
    mean = mean.view(1,-1, 1, 1)
    std = std.view(1,-1, 1, 1)
    return (data.to(device)-mean.to(device))/std.to(device)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class LearnableFNetBlock(nn.Module):
    def __init__(self, dim, patches):
        super().__init__()
        self.projection = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x, epoch):
        Freq = torch.fft.fft(torch.fft.fft(x.permute(0, 2, 1), dim=-1), dim=-2)

        b, patches, c = Freq.shape
        D_0 = patches // 2 + (patches // 8 - patches // 2) * (epoch / 60)
        lowpass_filter_l = torch.exp(
            -0.5 * torch.square(torch.linspace(0, patches // 2 - 1, patches // 2) / (patches // 8))).view(1,
                                                                                                          patches // 2,
                                                                                                          1).cuda()
        lowpass_filter_r = torch.flip(
            torch.exp(-0.5 * torch.square(torch.linspace(1, patches // 2, patches // 2) / (patches // 8))).view(1,
                                                                                                                patches // 2,
                                                                                                                1).cuda(),
            [1])
        lowpass_filter = torch.concat((lowpass_filter_l, lowpass_filter_r), dim=1)

        low_Freq = Freq * lowpass_filter
        lowFreq_feature = torch.fft.ifft(torch.fft.ifft(low_Freq, dim=-2), dim=-1).real
        weights = 0.5 * torch.sigmoid(self.projection(x).permute(0, 2, 1).mean(dim=1)).unsqueeze(dim=1) + 0.5

        out = weights * lowFreq_feature + (1 - weights) * (x.permute(0, 2, 1) - lowFreq_feature)

        return out.permute(0, 2, 1)

class WideResNet_FPCM(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, norm = False, mean = None, std = None):
        super(WideResNet_FPCM, self).__init__()
        self.num_classes = num_classes
        self.norm = norm
        self.mean = mean
        self.std = std
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # self.norm = Norm_layer((0,0,0), (1,1,1))
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.attn1 = LearnableFNetBlock(nChannels[1], 1024)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.attn2 = LearnableFNetBlock(nChannels[2], 256)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.attn3 = LearnableFNetBlock(nChannels[3], 64)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], self.num_classes)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, epoch=120):
        if self.norm == True:
            x = Normalization(x, self.mean, self.std)
        out = self.conv1(x)

        out = self.block1(out)
        b, c, h, w = out.shape
        out = self.attn1(out.view(b, c, h * w), epoch)  # + out.view(b, c, h*w)

        out = self.block2(out.view(b, c, h, w))
        b, c, h, w = out.shape
        out = self.attn2(out.view(b, c, h * w), epoch)  # + out.view(b, c, h*w)

        out = self.block3(out.view(b, c, h, w))
        b, c, h, w = out.shape
        out = self.attn3(out.view(b, c, h * w), epoch)  # + out.view(b, c, h*w)

        out = self.relu(self.bn1(out.view(b, c, h, w)))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def WRN28_10_FPCM(Num_class=10, Norm=False, norm_mean=None, norm_std=None):
    return WideResNet_FPCM(num_classes=Num_class, depth=34, widen_factor=10, norm=Norm, mean=norm_mean, std=norm_std)
