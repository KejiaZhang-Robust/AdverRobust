import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import Any, Tuple
import numpy as np

from tqdm import tqdm
import os
import shutil
from typing import Tuple
from torch import Tensor
from torch.autograd import Variable
from utils_train import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#TODO: MART
def train_adversarial_MART(net: nn.Module, epoch: int, train_loader: DataLoader, optimizer: Optimizer,
          config: Any, beta=6.0) -> Tuple[float, float]:
    print('\n[ Epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_bar = tqdm(total=len(train_loader), desc=f'>>')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        kl = nn.KLDivLoss(reduction='none')
        adv_inputs = pgd_attack(net, inputs, targets, config.Train.clip_eps / 255.,
                                config.Train.fgsm_step / 255., config.Train.pgd_train)

        optimizer.zero_grad()
        logits = net(inputs)
        logits_adv = net(adv_inputs)
        adv_probs = F.softmax(logits_adv, dim=1)

        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == targets, tmp1[:, -2], tmp1[:, -1])
        loss_adv = F.cross_entropy(logits_adv, targets) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
        nat_probs = F.softmax(logits, dim=1)
        true_probs = torch.gather(nat_probs, 1, (targets.unsqueeze(1)).long()).squeeze()
        loss_robust = (1.0 / len(inputs)) * torch.sum(
            torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
        loss = loss_adv + float(beta) * loss_robust
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = logits_adv.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_bar.set_postfix(train_acc=round(100. * correct / total, 2), loss=loss.item())
        train_bar.update()
    train_bar.close()
    print('Total benign train accuarcy:', 100. * correct / total)
    print('Total benign train loss:', train_loss)

    return 100. * correct / total, train_loss

#TODO: FSR CVPR-2023 "Feature Separation and Recalibration for Adversarial Robustness"
def train_adversarial_fsr(net: nn.Module, epoch: int, train_loader: DataLoader, optimizer: Optimizer,
          config: Any) -> Tuple[float, float]:
    print('\n[ Epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    train_bar = tqdm(total=len(train_loader), desc=f'>>')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inputs = pgd_attack(net, inputs, targets, config.Train.clip_eps / 255.,
                                config.Train.fgsm_step / 255., config.Train.pgd_train)
        
        adv_outputs, adv_r_outputs, adv_nr_outputs, adv_rec_outputs = net(adv_inputs, True)
        adv_labels = get_pred(adv_outputs, targets)

        adv_cls_loss = criterion(adv_outputs, targets)
        
        r_loss = torch.tensor(0.).to(device)
        if not len(adv_r_outputs) == 0:
            for r_out in adv_r_outputs:
                r_loss += criterion(r_out, targets)
            r_loss /= len(adv_r_outputs)

        nr_loss = torch.tensor(0.).to(device)
        if not len(adv_nr_outputs) == 0:
            for nr_out in adv_nr_outputs:
                nr_loss += criterion(nr_out, adv_labels)
            nr_loss /= len(adv_nr_outputs)
        sep_loss = r_loss + nr_loss

        rec_loss = torch.tensor(0.).to(device)
        if not len(adv_rec_outputs) == 0:
            for rec_out in adv_rec_outputs:
                rec_loss += criterion(rec_out, targets)
            rec_loss /= len(adv_rec_outputs)

        loss = adv_cls_loss + sep_loss + rec_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = adv_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_bar.set_postfix(loss=round(loss.item(),2), train_acc=round(100. * correct / total, 2))
        train_bar.update()
    train_bar.close()
    print('Total benign train accuarcy:', 100. * correct / total)
    print('Total benign train loss:', train_loss)

    return 100. * correct / total, train_loss

#TODO: ICCV-2023 "Towards Building More Robust Models with Frequency Bias"
def train_adversarial_fpcm(net: nn.Module, epoch: int, train_loader: DataLoader, optimizer: Optimizer,
          config: Any) -> Tuple[float, float]:
    print('\n[ Epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    train_bar = tqdm(total=len(train_loader), desc=f'>>')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inputs = pgd_attack(net, inputs, targets, config.Train.clip_eps / 255.,
                                config.Train.fgsm_step / 255., config.Train.pgd_train)

        optimizer.zero_grad()

        benign_outputs = net(adv_inputs, epoch)
        loss = criterion(benign_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_bar.set_postfix(train_acc=round(100. * correct / total, 2))
        train_bar.update()
    train_bar.close()
    print('Total benign train accuarcy:', 100. * correct / total)
    print('Total benign train loss:', train_loss)

    return 100. * correct / total, train_loss

#TODO: CVPR-2023 "The Enemy of My Enemy is My Friend: Exploring Inverse Adversaries for Improving Adversarial Training"
def pgd_invserse_attack_UIAT(model: nn.Module, x: Tensor, x_attack: Tensor, y: Tensor, epsilon: float, alpha: float, iters: int) -> Tensor:
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    criterion = nn.CrossEntropyLoss()
    logits_natural = model(x)

    for _ in range(iters):
        x_adv.requires_grad = True
        logits = model(x_adv)
        loss = criterion(logits, y) + F.kl_div(F.log_softmax(logits, dim=1),
                                F.softmax(logits_natural, dim=1),
                                reduction='batchmean') - F.kl_div(F.log_softmax(logits, dim=1),
                                F.softmax(model(x_attack), dim=1),
                                reduction='batchmean')
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() - alpha * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def train_adversarial_UIAT(net: nn.Module, epoch: int, train_loader: DataLoader, optimizer: Optimizer,
          config: Any, beta = 3.5) -> Tuple[float, float]:
    print('\n[ Epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    train_bar = tqdm(total=len(train_loader), desc=f'>>')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inputs = pgd_attack(net, inputs, targets, config.Train.clip_eps / 255.,
                                config.Train.fgsm_step / 255., config.Train.pgd_train)
        adv_invserse_inputs = pgd_invserse_attack_UIAT(net, inputs, adv_inputs, targets, config.Train.clip_eps / 255.,
                                config.Train.fgsm_step / 255., config.Train.pgd_train)
        optimizer.zero_grad()
        inv_outputs = net(adv_invserse_inputs)
        adv_outputs = net(adv_inputs)
        loss_logits = criterion(adv_outputs, targets)
        loss_1 = F.kl_div(F.log_softmax(inv_outputs, dim=1), F.softmax(adv_outputs, dim=1), reduction='batchmean')
        loss = loss_logits + beta*(loss_1)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = net(adv_inputs).max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_bar.set_postfix(acc=round(100. * correct / total, 2), loss=loss.item())
        train_bar.update()
    train_bar.close()

    return 100. * correct / total, train_loss

    
def get_pred(out, labels):
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)

    return adv_label