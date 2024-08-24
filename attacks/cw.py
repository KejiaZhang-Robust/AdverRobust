import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import Tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def evaluate_cw(net: nn.Module, test_loader: DataLoader, eps, step, iter) -> float:
    net.eval()
    adv_correct = 0
    total = 0
    progress_bar = tqdm(total=len(test_loader), desc='Testing-CW>>')
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

def evaluate_cw_transfer(net: nn.Module, test_net:nn.Module, test_loader: DataLoader, eps: int, step: int, iter: int) -> float:
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