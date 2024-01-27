import matplotlib.pyplot as plt
import numpy as np

# 数据
epsilon = [4, 8, 12, 16]
PGD_AT = [73.26, 56.07, 42.21, 33.97]
FMR_GC_AT = [74.87, 58.97, 45.67, 39.44]
FMR_GC_AT_1 = [73.96,57.03,37.61,24.28]
PGD_AT_1 = [72.03,54.21,34.23,17.28]
# 创建一个新的图形
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 7))

# 设置柱状图的宽度和位置
width = 0.3
x = np.arange(len(epsilon))

# 创建柱状图
rects1 = ax.bar(x - width/2, PGD_AT, width-0.02, label='PGD-AT', color='teal', alpha=0.8)
rects2 = ax.bar(x + width/2, FMR_GC_AT, width-0.02, label='FMR-GC-AT', color='darkorange', alpha=0.8)

# 创建折线图
ax.plot(x- width/2, PGD_AT, color='grey', marker='v', markeredgecolor = 'black', alpha=0.8, markersize=10, linewidth=2)
ax.plot(x+ width/2, FMR_GC_AT, color='grey', marker='v', markeredgecolor = 'black', alpha=0.8, markersize=10, linewidth=2)

# 添加柱状图上的标签
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height+0.1),
                    xytext=(0, 6),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14)

# 设置x轴和y轴的标签和标题
ax.set_ylim([30, 80])
ax.set_xlabel('$\epsilon$',fontsize=21)
ax.set_ylabel('Accuracy(%)', fontsize=21)
ax.set_title('PGD-10 Attack', fontsize=21)
ax.set_xticks(x)
ax.set_xticklabels(epsilon, fontsize=18)
ax.legend(loc='upper right', fontsize=18)
ax.grid(True, linestyle='--', alpha=0.5)

# 设置坐标轴刻度标签的大小
ax.tick_params(axis='both', which='major', labelsize=15)

# 创建柱状图
rects1 = ax2.bar(x - width/2, PGD_AT_1, width-0.02, label='PGD-AT', color='teal', alpha=0.8)
rects2 = ax2.bar(x + width/2, FMR_GC_AT_1, width-0.02, label='FMR-GC-AT', color='darkorange', alpha=0.8)

# 创建折线图
ax2.plot(x- width/2, PGD_AT_1, color='grey', marker='v', markeredgecolor = 'black', alpha=0.8, markersize=10, linewidth=2)
ax2.plot(x+ width/2, FMR_GC_AT_1, color='grey', marker='v', markeredgecolor = 'black', alpha=0.8, markersize=10, linewidth=2)

# 添加柱状图上的标签
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax2.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height+0.1),
                    xytext=(0, 4),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14)

# 设置x轴和y轴的标签和标题
ax2.set_ylim([15, 80])
ax2.set_xlabel('$\epsilon$',fontsize=21)
ax2.set_ylabel('Accuracy(%)', fontsize=21)
ax2.set_title('PGD-100 Attack', fontsize=21)
ax2.set_xticks(x)
ax2.set_xticklabels(epsilon, fontsize=18)
ax2.legend(loc='upper right', fontsize=18)
ax2.grid(True, linestyle='--', alpha=0.5)

# 设置坐标轴刻度标签的大小
ax2.tick_params(axis='both', which='major', labelsize=16)


# 显示图形
plt.tight_layout()
# plt.savefig('eplison_GNN.pdf', format='pdf')
plt.show()
