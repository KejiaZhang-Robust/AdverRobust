import torch.backends.cudnn as cudnn
from models import *
from utils_test import evaluate_normal, evaluate_pgd, evaluate_autoattack
from easydict import EasyDict
import yaml
import logging
import os
import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from models import *

# # 示例输入张量
# tensor = torch.randn((5, 256,1,1)).squeeze()  # (n, d)
# # 初始化相似度矩阵
# similarity_matrix = torch.zeros((tensor.size(0), tensor.size(0)))  # (n, n)

# # 计算相似度矩阵
# for i in range(tensor.size(0)):
#     for j in range(i+1, tensor.size(0)):
#         similarity = F.cosine_similarity(tensor[i].unsqueeze(0), tensor[j].unsqueeze(0), dim=1)
#         similarity_matrix[i, j] = similarity
#         similarity_matrix[j, i] = similarity

# print("Similarity Matrix:")
# print(torch.sum(similarity_matrix)/2)

import torch
from torchsummary import summary

# # 定义你的模型
# model = ResNet18_F()

# # 将模型移动到适当的设备（如GPU）
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # 使用torchsummary库来查看模型的计算复杂度指标
# summary(model, input_size=(3, 32, 32))


def CW_loss2(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    # 最大的预测概率是否等于y

    loss_value = -(x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

# x_attack_true = torch.where(torch.argmax(logit_x, dim=1) != y, torch.tensor(1.0), torch.tensor(0.0))
    # x_attack_defense_true = (x_attack_true > 0).int().float()
    # x_attack_defense_false = 1 - x_attack_defense_true
    # loss_x = F.cross_entropy(logit_x, y, reduction='none')
    # loss_x_topk2 = F.cross_entropy(logit_x, torch.topk(logit_x, 2, dim=1)[1][:, 1], reduction='none')
    # loss = loss_x * x_attack_defense_false-loss_x_topk2 * x_attack_defense_false

def CW_loss_x(logit_x, y):
    logitx_value,logitx_indices = torch.topk(logit_x, 2, dim=1)
    ind = (logitx_indices[:,-2] == y).float()
    loss_value = -(logit_x[torch.arange(logit_x.shape[0]), y] - logitx_value[:,-1]*ind-logitx_value[:,-2]*(1.-ind))

    return loss_value.mean()

def Union_loss_inverse(logit_x, y, logit_x_inverse):
    logitx_value,logitx_indices = torch.topk(logit_x, 2, dim=1)
    logitix_value,logitix_indices = torch.topk(logit_x_inverse, 2, dim=1)
    x_attack_true = (logitx_indices[:,-2] != y).float()
    ix_defense_true = (logitix_indices[:,-2] == y).float()
    enhance_attack_id = (x_attack_true > 0 &  ix_defense_true>0).int().float()
    loss_value = -(x[torch.arange(x.shape[0]), y] - logitx_value[:,-1]*enhance_attack_id-logitx_value[:,-2]*(1.-enhance_attack_id))
    return loss_value.mean()


x = torch.tensor([[0.1,0.2,0.3],[0.3,0.1,0.2]])
x_sorted, ind_sorted = x.sort(dim=1)
print(ind_sorted[:,-1])
y = torch.tensor([2,1])
ind = (ind_sorted[:, -1] == y).float()
print((ind_sorted[:, -1] == y).float())
print(x[torch.arange(x.shape[0]), y]) #输出 正确的 概率，而不是 标签
print(x_sorted[:, -2] * ind) #输出 第二大的概率
print(x_sorted[:, -1] * (1. - ind)) #输出 最大的概率
print(-(x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)))
# 第一项 是 预测正确的，第二项 没有 预测正确

logitx_value,logitx_indices = torch.topk(x, 2, dim=1)

print(logitx_indices[:,-1]) #输出 第二的概率的标签
print(logitx_indices[:,-2]) #输出 第一的概率的标签
ind1 = (logitx_indices[:,-2]==y).float()
print((logitx_indices[:,-2]==y).float())
print(-(x[torch.arange(x.shape[0]), y]-logitx_value[:,-1]*ind1-logitx_value[:,-2]*(1.-ind1)))
print(CW_loss_x(x,y))