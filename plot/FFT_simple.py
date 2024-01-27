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
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, norm = False, mean = None, std = None):
        super(WideResNet_FFT, self).__init__()
        self.norm = norm
        self.mean = mean
        self.std = std
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
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

    def forward(self, x, k):
        if self.norm == True:
            x = Normalization(x, self.mean, self.std)
        out = self.conv1(x)
        out = reconstruct_feature_pytorch(out, k)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


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