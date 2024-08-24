import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import Tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# EOT-PGD attack
def eot_pgd_attack(model: nn.Module, x: Tensor, y: Tensor, epsilon: float, alpha: float, iters: int, eot: int) -> Tensor:
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    criterion = nn.CrossEntropyLoss()

    for _ in range(iters):
        grad = torch.zeros_like(x_adv)
        for _ in range(eot):
            x_adv.requires_grad = True
            logits = model(x_adv)
            loss = criterion(logits, y)
            grad += torch.autograd.grad(loss, x_adv)[0].detach()
            x_adv = x_adv.detach()

        grad /= eot
        grad = grad.sign()
        x_adv = x_adv + alpha * grad
        
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = x_adv.detach()
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def eot_pgd_attack_l2(model: nn.Module, x: Tensor, y: Tensor, epsilon: float, alpha: float, iters: int, eot: int) -> Tensor:
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    criterion = nn.CrossEntropyLoss()

    for _ in range(iters):
        grad = torch.zeros_like(x_adv)
        for _ in range(eot):
            x_adv.requires_grad = True
            logits = model(x_adv)
            loss = criterion(logits, y)
            grad += torch.autograd.grad(loss, x_adv)[0].detach()
            x_adv = x_adv.detach()

        grad /= eot
        grad = grad.sign()
        x_adv = x_adv + alpha * grad
        
        delta = x_adv - x
        delta_norms = torch.norm(delta.view(x.shape[0], -1), p=2, dim=1)
        factor = epsilon / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)
        
        x_adv = x + delta
        x_adv = x_adv.detach()
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def evaluate_eot_pgd(net: nn.Module, test_loader: DataLoader, eps, step, iter, eot=2) -> float:
    net.eval()
    adv_correct = 0
    total = 0
    progress_bar = tqdm(total=len(test_loader), desc='Testing-PGD>>')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = eot_pgd_attack(net,inputs,targets, eps/255., step/255., iter, eot=eot)
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

def evaluate_eot_pgd_l2(net: nn.Module, test_loader: DataLoader, eps, step, iter, eot=2) -> float:
    net.eval()
    adv_correct = 0
    total = 0
    progress_bar = tqdm(total=len(test_loader), desc='Testing-PGD>>')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = eot_pgd_attack_l2(net,inputs,targets, eps/255., step/255., iter, eot=eot)
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