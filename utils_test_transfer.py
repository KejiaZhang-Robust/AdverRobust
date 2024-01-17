import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import torch as ch
from torch.utils.data import Dataset
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import Any, Tuple

import torchvision.transforms as transforms
# from autoattack import AutoAttack
from tqdm import tqdm
import os
import shutil
from typing import Tuple
from torch import Tensor
import numpy as np
from autoattack import *
from utils_test import cw_Linf_attack, pgd_attack

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_pgd(net: nn.Module, test_net:nn.Module, test_loader: DataLoader, eps: int, step: int, iter: int) -> float:
    net.eval()
    test_net.eval()
    adv_correct = 0
    total = 0
    progress_bar = tqdm(total=len(test_loader), desc='Testing-PGD>>')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = pgd_attack(net, inputs, targets, eps/255., step/255., iter)
        with torch.no_grad():
            adv_outputs = test_net(adv)
        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()
        if total % 100 == 0:
            progress_bar.set_postfix(test_pgd_acc=round(100. * adv_correct / total, 2))
        progress_bar.update(1)  # update bar
    progress_bar.close()  # close bar
    adv_acc = 100. * adv_correct / total
    return adv_acc

def evaluate_cw(net: nn.Module, test_net:nn.Module, test_loader: DataLoader, eps: int, step: int, iter: int) -> float:
    net.eval()
    test_net.eval()
    adv_correct = 0
    total = 0
    progress_bar = tqdm(total=len(test_loader), desc='Testing-PGD>>')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = cw_Linf_attack(net, inputs, targets, eps/255., step/255., iter)
        with torch.no_grad():
            adv_outputs = test_net(adv)
        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()
        if total % 100 == 0:
            progress_bar.set_postfix(test_pgd_acc=round(100. * adv_correct / total, 2))
        progress_bar.update(1)  # update bar
    progress_bar.close()  # close bar
    adv_acc = 100. * adv_correct / total
    return adv_acc

def evaluate_normal(test_net: nn.Module, test_loader: DataLoader) -> float:
    test_net.eval()
    benign_correct = 0
    total = 0
    total_len_test = len(test_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader),total=total_len_test):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            outputs = test_net(inputs)

            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

    test_acc = 100. * benign_correct / total

    return test_acc

def test_adv_auto(net: nn.Module, test_net: nn.Module, test_loader: DataLoader, eps: int, attacks_run: list):
    net.eval()
    test_net.eval()
    benign_correct = 0
    total = 0
    autoattack = AutoAttack(net, norm='Linf', eps=eps/255., seed=1,
                            attacks_to_run=attacks_run, version='custom')
    autoattack.apgd.n_restarts = 2
    autoattack.fab.n_restarts = 2
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv,_ = autoattack.run_standard_evaluation(inputs, targets, bs=250)
        with torch.no_grad():
            adv_outputs_batch = test_net(adv)
        _, predicted = adv_outputs_batch.max(1)
        benign_correct += predicted.eq(targets).sum().item()
    test_acc = 100. * benign_correct / total
    
    print(f"Autoattack have done! Accruracy{test_acc:.2f}")
    return test_acc


