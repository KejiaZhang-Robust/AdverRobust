import torch.backends.cudnn as cudnn
import torchvision

import torchvision.transforms as transforms

from tqdm import tqdm
import os
from torch import Tensor
from torch.utils.data import DataLoader

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

file_name = 'WRN32_10_AT'

tree_name = 'AT_PGD_10_1'

normalize_mean = torch.Tensor([0.4914, 0.4822, 0.4465]).to(device)
normalize_std = torch.Tensor([0.2023, 0.1994, 0.2010]).to(device)

class WideResNet_FFT(nn.Module):
    def __init__(self, num_classes, depth, widen_factor, norm = False,  dropout_rate=0.0):
        super(WideResNet_FFT, self).__init__()
        self.in_planes = 16
        self.norm = norm

        n = int((depth - 4) / 6)
        k = widen_factor

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(WideBasicBlock, 16*k, n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasicBlock, 32*k, n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasicBlock, 64*k, n, dropout_rate, stride=2)
        self.bn = nn.BatchNorm2d(64*k)
        self.linear = nn.Linear(64*k, num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x, k):
        if self.norm == True:
            x = Normalization(x, normalize_mean, normalize_std)
        out = F.relu(self.conv1(x))
        x = reconstruct_feature_pytorch(x, k)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def pgd_attack(model: nn.Module, x: Tensor, y: Tensor, epsilon: float, alpha: float, iters: int) -> Tensor:
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    criterion = nn.CrossEntropyLoss()

    for _ in range(iters):
        x_adv.requires_grad = True
        logits = model(x_adv)
        loss = criterion(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()


net = WideResNet(num_classes=10, depth=32, widen_factor=10, norm=True, mean=normalize_mean, std=normalize_std)
net = net.to(device)
net = torch.nn.DataParallel(net)  # parallel GPU
cudnn.benchmark = True

test_net = WideResNet_FFT(num_classes=10, depth=32, widen_factor=10,norm=True)
test_net = test_net.to(device)
test_net = torch.nn.DataParallel(test_net)  # parallel GPU

dataset = 'CIFAR10'
if not os.path.isdir(os.path.join('./FFT', tree_name)):
    os.mkdir(os.path.join('./FFT', tree_name))
check_path = os.path.join('./checkpoint', dataset, file_name)

checkpoint_best = torch.load(os.path.join(check_path, 'checkpoint.pth.tar'))
net.load_state_dict(checkpoint_best['state_dict'])
test_net.load_state_dict(checkpoint_best['state_dict'])
net.eval()
test_net.eval()

indices = list(range(1000))
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
subset = torch.utils.data.Subset(test_dataset, indices)
test_loader = torch.utils.data.DataLoader(subset, batch_size=100, shuffle=False, num_workers=4)


# TODO: adversarial was pertured in the input images.
eps_num = [0, 4, 8, 12]
for e_iter in eps_num:
    k_list = [i for i in range(0,32*32,1)]
    Re_bar = tqdm(total=len(k_list), desc=f'eps:{e_iter}>>')
    for k in k_list:
        benign_correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if e_iter == 0:
                inputs = inputs
            else:
                inputs = pgd_attack(net, inputs, targets, e_iter / 255., 2 / 255., 10)
            outputs = test_net(inputs,k)
            total += targets.size(0)
            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()
        Re_bar.set_postfix(k=k, acc=round(100. * benign_correct / total, 2))
        Re_bar.update(1)
        with open(os.path.join('./FFT', tree_name, 'eps' + str(e_iter) + '.txt'), "a+") as f:
            f.write(f'{100. * benign_correct / total:.2f}\n')
        f.close()
    Re_bar.close()
    benign_correct = 0
    total = 0
    progress_bar = tqdm(total=len(test_loader), desc='Testing>>')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if e_iter == 0:
            inputs = inputs
        else:
            inputs = pgd_attack(net, inputs, targets, e_iter / 255., 2 / 255., 10)
        outputs = net(inputs)
        total += targets.size(0)
        _, predicted = outputs.max(1)
        benign_correct += predicted.eq(targets).sum().item()
        progress_bar.update(1)
    progress_bar.close()
    print(f'eps:{e_iter}---test_acc{100. * benign_correct / total:.2f}\n')
    with open(os.path.join('./FFT', tree_name, 'attack_acc_net'), "a+") as f:
        f.write(f'eps:{e_iter}---test_acc{100. * benign_correct / total:.2f}\n')
    f.close()