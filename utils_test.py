import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import Tensor
from autoattack import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Normalization(data, mean, std):
    mean = mean.view(1,-1, 1, 1)
    std = std.view(1,-1, 1, 1)
    return (data.to(device)-mean.to(device))/std.to(device)

# PGD attack
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

def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    loss_value = -(x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

def cw_Linf_attack(model: nn.Module, x: Tensor, y: Tensor, epsilon: float, alpha: float, iters: int) -> Tensor:
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(iters):
        x_adv.requires_grad = True
        logits = model(x_adv)
        loss = CW_loss(logits, y)
        loss.backward()
        grad = x_adv.grad.detach()

        x_adv = x_adv.detach() + alpha * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def evaluate_pgd(net: nn.Module, test_loader: DataLoader, eps, step, iter) -> float:
    net.eval()
    adv_correct = 0
    total = 0
    progress_bar = tqdm(total=len(test_loader), desc='Testing-PGD>>')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = pgd_attack(net,inputs,targets, eps/255., step/255., iter)
        with torch.no_grad():
            adv_outputs = net(adv)
        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()
        if total % 100 == 0:
            progress_bar.set_postfix(test_pgd_acc=round(100. * adv_correct / total, 2))
        progress_bar.update(1)  # update bar
    progress_bar.close()  # close bar
    adv_acc = 100. * adv_correct / total
    return adv_acc

def evaluate_cw(net: nn.Module, test_loader: DataLoader, eps, step, iter) -> float:
    net.eval()
    adv_correct = 0
    total = 0
    progress_bar = tqdm(total=len(test_loader), desc='Testing-PGD>>')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = cw_Linf_attack(net, inputs, targets, eps/255, step/255, iter)
        with torch.no_grad():
            adv_outputs = net(adv)
        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()
        if total % 100 == 0:
            progress_bar.set_postfix(test_pgd_acc=round(100. * adv_correct / total, 2))
        progress_bar.update(1)  # update bar
    progress_bar.close()  # close bar
    adv_acc = 100. * adv_correct / total
    return adv_acc

def evaluate_normal(net: nn.Module, test_loader: DataLoader) -> float:
    net.eval()
    benign_correct = 0
    total = 0
    total_len_test = len(test_loader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader),total=total_len_test):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

    test_acc = 100. * benign_correct / total

    return test_acc

def evaluate_autoattack(net: nn.Module, test_loader: DataLoader, eps: int, attacks_run: list) -> float:
    net.eval()

    autoattack = AutoAttack(net, norm='Linf', eps=eps/255., seed=1,
                            attacks_to_run=attacks_run, version='custom')
    autoattack.apgd.n_restarts = 2
    autoattack.fab.n_restarts = 2
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    #TODO: Modify return value of run_standard_evaluation
    x_adv, robust_accuracy = autoattack.run_standard_evaluation((x_test).to(device), y_test.to(device))

    print(f"Autoattack have done! Accruracy{robust_accuracy*100.:.2f}")

    return robust_accuracy*100.

def test_adv(net,adversary,test_loader):
    net.eval()
    adv_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = adversary.perturb(inputs, targets)
        adv_outputs = net(adv)
        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()

    adv_acc = 100. * adv_correct / total
    return adv_acc


