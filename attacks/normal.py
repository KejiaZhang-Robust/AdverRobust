import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import Tensor
from autoattack import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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