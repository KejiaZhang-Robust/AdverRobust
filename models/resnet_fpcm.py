import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Normalization(data, mean, std):
    mean = mean.view(1,-1, 1, 1)
    std = std.view(1,-1, 1, 1)
    return (data.to(device)-mean.to(device))/std.to(device)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class LearnableFNetBlock(nn.Module):
    def __init__(self, dim, patches):
        super().__init__()
        self.projection = nn.Conv1d(dim, dim, kernel_size=1, groups=dim)     
        self.learnable_freq_control = nn.Parameter(torch.ones(dim)) # Deprecated                             

    def forward(self, x, epoch):
        
        Freq = torch.fft.fft(torch.fft.fft(x.permute(0,2,1), dim=-1), dim=-2)

        b, patches, c = Freq.shape
        D_0 = patches // 2 + (patches // 8 - patches // 2) * (epoch / 120)
        lowpass_filter_l = torch.exp(-0.5 * torch.square(torch.linspace(0, patches // 2 - 1, patches // 2).unsqueeze(1).repeat(1,c).cuda() / (D_0))).view(1, patches // 2, c).cuda()
        lowpass_filter_r = torch.flip(torch.exp(-0.5 * torch.square(torch.linspace(1, patches // 2 , patches // 2).unsqueeze(1).repeat(1,c).cuda() / (D_0))).view(1, patches // 2, c).cuda(), [1])
        lowpass_filter = torch.concat((lowpass_filter_l, lowpass_filter_r), dim=1)
        
        low_Freq = Freq * lowpass_filter
        lowFreq_feature = torch.fft.ifft(torch.fft.ifft(low_Freq, dim=-2), dim=-1).real

        weights = 0.5 * torch.sigmoid(self.projection(x).permute(0,2,1).mean(dim=1)).unsqueeze(dim=1) + 0.5
        out = weights * lowFreq_feature + (1 - weights) * (x.permute(0,2,1) - lowFreq_feature)

        return out.permute(0,2,1)

    
class ResNet18_FNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm = False, mean = None, std = None):
        super(ResNet18_FNet, self).__init__()
        self.norm = norm
        self.mean = mean
        self.std = std
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.attn1 = LearnableFNetBlock(64, 1024)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.attn2 = LearnableFNetBlock(128, 256)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.attn3 = LearnableFNetBlock(256, 64)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.attn4 = LearnableFNetBlock(512, 16)
        self.linear = nn.Linear(512*block.expansion*4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, epoch=120):
        if self.norm == True:
            x = Normalization(x, self.mean, self.std)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        b, c, h, w = out.shape
        out = self.attn1(out.view(b, c, h*w), epoch) 

        out = self.layer2(out.view(b, c, h, w))
        b, c, h, w = out.shape
        out = self.attn2(out.view(b, c, h*w), epoch) 

        out = self.layer3(out.view(b, c, h, w))
        b, c, h, w = out.shape
        out = self.attn3(out.view(b, c, h*w), epoch) 

        out = self.layer4(out.view(b, c, h, w))
        b, c, h, w = out.shape
        out = self.attn4(out.view(b, c, h*w), epoch) 
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18_fpcm(Num_class=10, Norm=False, norm_mean=None, norm_std=None):
    return ResNet18_FNet(BasicBlock, [2,2,2,2], num_classes=Num_class, norm=Norm, mean=norm_mean, std=norm_std)
# def train_adversarial_fpcm(net: nn.Module, epoch: int, train_loader: DataLoader, optimizer: Optimizer,
#           config: Any) -> Tuple[float, float]:
#     print('\n[ Epoch: %d ]' % epoch)
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     criterion = nn.CrossEntropyLoss()
#     train_bar = tqdm(total=len(train_loader), desc=f'>>')
#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         adv_inputs = pgd_attack(net, inputs, targets, config.Train.clip_eps / 255.,
#                                 config.Train.fgsm_step / 255., config.Train.pgd_train)

#         optimizer.zero_grad()

#         benign_outputs = net(adv_inputs, epoch)
#         loss = criterion(benign_outputs, targets)
#         loss.backward()

#         optimizer.step()
#         train_loss += loss.item()
#         _, predicted = benign_outputs.max(1)

#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()
#         train_bar.set_postfix(train_acc=round(100. * correct / total, 2))
#         train_bar.update()
#     train_bar.close()
#     print('Total benign train accuarcy:', 100. * correct / total)
#     print('Total benign train loss:', train_loss)

#     return 100. * correct / total, train_loss