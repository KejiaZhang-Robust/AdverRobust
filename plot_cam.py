import torch
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# # 加载输入图像和特征图
# input_tensor = torch.load('images_input.pt')
# feature_map_tensor = torch.load('images_conv1.pt')

# # 转换为 NumPy 数组
# input_array = input_tensor.detach().numpy()
# feature_map_array = feature_map_tensor.detach().numpy()

# # 绘制输入图像
# plt.figure(figsize=(10, 4))
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(input_array[i].transpose(1, 2, 0))
#     plt.axis('off')
# plt.suptitle('Input Images')
# plt.show()

# # 绘制特征图
# plt.figure(figsize=(10, 4))
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(feature_map_array[i, 0], cmap='jet')
#     plt.axis('off')
# plt.suptitle('Feature Maps')
# plt.show()



from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet18

class FFT_1D_NonLocal_Means(nn.Module):
    def __init__(self, K) -> None:
        super(FFT_1D_NonLocal_Means, self).__init__()
        self.K = K

    def forward(self, x):
        feature_maps = x
        batch_size, num_features, height, width = feature_maps.shape
        feature_maps = feature_maps.view(batch_size, num_features, -1)
        feature_maps_fft = torch.fft.fft(feature_maps, dim=2)
        freqs = torch.fft.fftfreq(feature_maps_fft.shape[-1])

        _, idx = torch.topk(freqs.abs(), self.K, largest=False)  

        mask = torch.zeros_like(feature_maps_fft, dtype=torch.bool)
        mask[:, :, idx] = 1

        feature_maps_fft_fil = feature_maps_fft.where(mask, torch.zeros_like(feature_maps_fft))
        feature_maps_recon = torch.fft.ifft(feature_maps_fft_fil, dim=2).real 
        
        return feature_maps_recon.view(batch_size, num_features, height, width)
    
class ParentModule(nn.Module):
    def __init__(self):
        super(ParentModule, self).__init__()
        self.fft_module = FFT_1D_NonLocal_Means(int(32*32*0.75))

class MixUpResNet(nn.Module):
    def __init__(self,original_model):
        super(MixUpResNet, self).__init__()
        self.features = nn.Sequential()
        self.FFT = ParentModule()
        for name, module in original_model.named_children():
            if name == 'fc':
                break
            else:
                self.features.add_module(name, module)
            if name == 'relu':
                for name_1, module_1 in self.FFT.named_children():
                    self.features.add_module(name_1, module_1)
        self.last_linear = original_model.fc
    def forward(self,x):
        for layer in self.features:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x



model = resnet18(pretrained=True)
target_layers = [model.layer4[-1]]
# 读取JPEG图像
from PIL import Image
import torch,torchvision
image_path = 'panda.JPEG'
image = Image.open(image_path)
# 转换为张量
transform = torchvision.transforms.ToTensor()
input_tensor = transform(image).unsqueeze(0)

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers)

targets = [ClassifierOutputTarget(388)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

input_array = input_tensor.squeeze().permute(1, 2, 0).numpy()
input_array = (input_array - input_array.min()) / (input_array.max() - input_array.min())

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(input_array, grayscale_cam, use_rgb=True)
print(np.shape(visualization))
# 展示可视化结果
plt.imshow(visualization)
plt.axis('off')
plt.show()