import torch.backends.cudnn as cudnn
import torchvision

import torchvision.transforms as transforms

from tqdm import tqdm
import os
from torch import Tensor
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from models import *
from utils_test import evaluate_normal, evaluate_pgd, evaluate_autoattack, evaluate_cw
from easydict import EasyDict
import yaml
import logging
import os
from models import *
from utils import create_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('configs_test.yml') as f:
    config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

net = WRN34_10(Num_class=config.DATA.num_class)
test_net = WRN34_10_F(Num_class=config.DATA.num_class)

file_name = config.Operation.Prefix
data_set = config.DATA.Data
check_path = os.path.join('./checkpoint', data_set, file_name)
if not os.path.isdir(check_path):
    os.mkdir(check_path)

norm_mean = torch.tensor(config.DATA.mean).to(device)
norm_std = torch.tensor(config.DATA.std).to(device)
if config.Operation.Method == 'AT':
    net.Norm = True
    net.norm_mean = norm_mean
    net.norm_std = norm_std
    Data_norm = False
    test_net.Norm = True
    test_net.norm_mean = norm_mean
    test_net.norm_std = norm_std
else:
    net.Norm = False
    Data_norm = True

_, test_loader = create_dataloader(data_set, Norm=Data_norm)

net = net.to(device)
net = torch.nn.DataParallel(net)  # parallel GPU
test_net = test_net.to(device)
test_net = torch.nn.DataParallel(test_net)  # parallel GPU
cudnn.benchmark = True

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

tree_name = 'PGD'

checkpoint_best = torch.load(os.path.join(check_path, 'model_best.pth.tar'))
net.load_state_dict(checkpoint_best['state_dict'])
test_net.load_state_dict(checkpoint_best['state_dict'])
net.eval()
test_net.eval()

# TODO: adversarial was pertured in the input images.
eps_num = [0, 4, 8, 12]
for e_iter in eps_num:
    new_inputs_list = []
    new_targets_list = []
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if e_iter == 0:
            new_inputs_list.append(inputs)
        else:
            inputs = pgd_attack(net, inputs, targets, e_iter / 255., 2 / 255., 10)
            new_inputs_list.append(inputs)
        new_targets_list.append(targets)
    new_inputs = torch.cat(new_inputs_list, dim=0)
    new_targets = torch.cat(new_targets_list, dim=0)
    new_dataset = torch.utils.data.TensorDataset(new_inputs, new_targets)
    new_test_loader = torch.utils.data.DataLoader(new_dataset, batch_size=100, shuffle=True)
    progress_bar = tqdm(total=len(new_test_loader), desc='Testing>>')
    benign_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(new_test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = test_net(inputs)
        total += targets.size(0)
        _, predicted = outputs.max(1)
        benign_correct += predicted.eq(targets).sum().item()
        progress_bar.update(1)
    progress_bar.close()
    print(f'eps:{e_iter}---test_acc{100. * benign_correct / total:.2f}\n')
    with open(os.path.join('./FFT', tree_name, 'attack_acc_net'), "a+") as f:
        f.write(f'eps:{e_iter}---test_acc{100. * benign_correct / total:.2f}\n')
    f.close()
    k_list = [i for i in range(0,32*32,1)]
    Re_bar = tqdm(total=len(k_list), desc=f'eps:{e_iter}>>')
    for k in k_list:
        benign_correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(new_test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
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
