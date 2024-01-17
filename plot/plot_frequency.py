import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

def savitzky_golay(a, window_size=5, order=3):
    return savgol_filter(a, window_size, order)

def generate_gradient_color(cmap, n):
    gradient = np.linspace(0, 1, n)
    colors = [cmap(x) for x in gradient]
    return colors

# 创建一个包含两个子图的图形窗口
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,7))

Prex = 'WRN34_10'

# [95.87,33.48,15.86,9.42] [85.62, 70.24, 50.57, 35.69]
eps_adv_without = [85.80, 73.27,56.38,41.31]

record_path = os.path.join(Prex)

with open(os.path.join(record_path,'eps0.txt')) as f:
    lines = f.readlines()
    eps0 = [float(line.strip()) for line in lines]

with open(os.path.join(record_path,'eps4.txt')) as f:
    lines = f.readlines()
    eps4 = [float(line.strip()) for line in lines]

with open(os.path.join(record_path,'eps8.txt')) as f:
    lines = f.readlines()
    eps8 = [float(line.strip()) for line in lines]

with open(os.path.join(record_path,'eps12.txt')) as f:
    lines = f.readlines()
    eps12 = [float(line.strip()) for line in lines]

k_values = np.arange(0,len(eps8),1)
total_range = len(eps0)
length_dot=len(eps0)
colors = generate_gradient_color(plt.cm.Blues, 8)
colors_1 = generate_gradient_color(plt.cm.Blues, 8)
color_eps0 = 'indianred'

ax1.plot(k_values/total_range, eps0, label='$\epsilon=0$', linewidth=2.5, color=color_eps0)
ax1.plot(k_values/total_range, eps4, label='$\epsilon=4$', linewidth=2.5, color=colors[3])
ax1.plot(k_values/total_range, eps8, label='$\epsilon=8$', linewidth=2.5, color=colors[5])
ax1.plot(k_values/total_range, eps12, label='$\epsilon=12$', linewidth=2.5, color=colors_1[7])

ax1.plot(k_values/total_range,[eps_adv_without[0] for i in range(length_dot)], linewidth=2.5, linestyle='--', color=color_eps0)
ax1.plot(k_values/total_range,[eps_adv_without[1] for i in range(length_dot)], linewidth=2.5, linestyle='--', color=colors[3])
ax1.plot(k_values/total_range,[eps_adv_without[2] for i in range(length_dot)], linewidth=2.5, linestyle='--', color=colors[5])
ax1.plot(k_values/total_range,[eps_adv_without[3] for i in range(length_dot)], linewidth=2.5, linestyle='--', color=colors_1[7])


ax1.fill_between(k_values/total_range, np.array(eps0)-1.6, np.array(eps0)+1.6, alpha=0.13, color=color_eps0)
ax1.fill_between(k_values/total_range, np.array(eps4)-1.6, np.array(eps4)+1.6, alpha=0.13, color=colors[3])
ax1.fill_between(k_values/total_range, np.array(eps8)-1.6, np.array(eps8)+1.6, alpha=0.13, color=colors[5])
ax1.fill_between(k_values/total_range, np.array(eps12)-1.6, np.array(eps12)+1.6, alpha=0.13, color=colors_1[7])

# 设置左边子图的标签和刻度字体大小
ax1.set_xlabel('Frequency Components', fontsize=28)
ax1.set_ylabel('ST Accuracy (%)', fontsize=28)
ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)

# 设置左边子图的坐标轴范围
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 100])


# plt.title('AT Recostruct signal on Images (PGD-10 attack)', fontsize=18)
plt.subplots_adjust(wspace=0.18)

Prex = 'HFDR'

eps_adv_without = [85.41,73.66,58.64,45.59]

record_path = os.path.join(Prex)

with open(os.path.join(record_path,'eps0.txt')) as f:
    lines = f.readlines()
    eps0 = [float(line.strip()) for line in lines]

with open(os.path.join(record_path,'eps4.txt')) as f:
    lines = f.readlines()
    eps4 = [float(line.strip()) for line in lines]

with open(os.path.join(record_path,'eps8.txt')) as f:
    lines = f.readlines()
    eps8 = [float(line.strip()) for line in lines]

with open(os.path.join(record_path,'eps12.txt')) as f:
    lines = f.readlines()
    eps12 = [float(line.strip()) for line in lines]

k_values = np.arange(0,len(eps8),1)
total_range = len(eps0)
length_dot=len(eps0)
# colors = generate_gradient_color(plt.cm.Blues, 10)
# colors_1 = generate_gradient_color(plt.cm.Blues, 10)
# color_eps0 = 'orangered'

ax2.plot(k_values/total_range, eps0, label='$\epsilon=0$', linewidth=2.5, color=color_eps0)
ax2.plot(k_values/total_range, eps4, label='$\epsilon=4$', linewidth=2.5, color=colors[3])
ax2.plot(k_values/total_range, eps8, label='$\epsilon=8$', linewidth=2.5, color=colors[5])
ax2.plot(k_values/total_range, eps12, label='$\epsilon=12$', linewidth=2.5, color=colors_1[7])

ax2.plot(k_values/total_range,[eps_adv_without[0] for i in range(length_dot)], linewidth=2.5, linestyle='--', color=color_eps0)
ax2.plot(k_values/total_range,[eps_adv_without[1] for i in range(length_dot)], linewidth=2.5, linestyle='--', color=colors[3])
ax2.plot(k_values/total_range,[eps_adv_without[2] for i in range(length_dot)], linewidth=2.5, linestyle='--', color=colors[5])
ax2.plot(k_values/total_range,[eps_adv_without[3] for i in range(length_dot)], linewidth=2.5, linestyle='--', color=colors_1[7])


ax2.fill_between(k_values/total_range, np.array(eps0)-1.6, np.array(eps0)+1.6, alpha=0.13, color=color_eps0)
ax2.fill_between(k_values/total_range, np.array(eps4)-1.6, np.array(eps4)+1.6, alpha=0.13, color=colors[3])
ax2.fill_between(k_values/total_range, np.array(eps8)-1.6, np.array(eps8)+1.6, alpha=0.13, color=colors[5])
ax2.fill_between(k_values/total_range, np.array(eps12)-1.6, np.array(eps12)+1.6, alpha=0.13, color=colors_1[7])

# 设置左边子图的标签和刻度字体大小
ax2.set_xlabel('Frequency Components', fontsize=28)
ax2.set_ylabel('AT Accuracy (%)', fontsize=28)
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)

# 设置左边子图的坐标轴范围
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 100])
ax2.legend(loc='lower right', fontsize=20)
ax1.grid(True, linestyle='--', alpha=0.5)
ax2.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
# plt.savefig(os.path.join(record_path,'figure_4.pdf'), format='pdf')
plt.show()
