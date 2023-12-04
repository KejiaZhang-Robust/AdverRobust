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

# 定义你的模型
model = ResNet18_F()

# 将模型移动到适当的设备（如GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 使用torchsummary库来查看模型的计算复杂度指标
summary(model, input_size=(3, 32, 32))
