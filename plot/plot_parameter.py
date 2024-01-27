import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np

lamda = [0, 0.1, 0.25, 0.50, 0.75, 1]
lamda_clean = [80.16, 81.27, 80.40, 79.91, 79.86, 79.75]
lamda_PGD = [53.92, 54.69, 54.79, 55.02, 55.04, 55.12]

beta = [0, 0.1, 0.25, 0.50, 0.75, 1]
beta_clean = [80.16, 81.27, 81.56, 81.87, 81.52, 81.32]
beta_PGD = [53.92, 54.69, 54.48, 54.47, 54.53, 54.22]

fig, (ax1,ax3) = plt.subplots(1, 2, figsize=(18,7))

ax1.plot(lamda, lamda_clean, label='Clean ($\lambda$)',linewidth=2.5, marker='o', alpha=0.8,markersize=10, color='indianred')
ax1.set_xlabel('$\lambda$', fontsize=26)
ax1.set_ylabel('Accuracy(%)', fontsize=26)
ax1.tick_params(axis='y', labelsize=22)

ax2 = ax1.twinx()
ax2.plot(lamda, lamda_PGD, label='PGD-10 ($\lambda$)', linewidth=2.5, marker='v', alpha=0.8,markersize=10,color='teal')
# ax2.set_ylabel('PGD-10 Accuracy(%)', color='teal', fontsize=20)
ax2.tick_params(axis='y', labelsize=22)

ax3.plot(beta, beta_clean, label=r'Clean ($\beta$)', linewidth=2.5, marker='*', alpha=0.8,markersize=10,color='seagreen')
ax3.set_xlabel(r'$\beta$', fontsize=26)
ax3.set_ylabel('Accuracy(%)', fontsize=26)
ax3.tick_params(axis='y', labelsize=22)

ax4 = ax3.twinx()
ax4.plot(beta, beta_PGD, label=r'PGD-10 ($\beta$)', linewidth=2.5, marker='s', alpha=0.8,markersize=8,color='darkgoldenrod')
# ax4.set_ylabel('PGD-10 Accuracy(%)', color='darkgoldenrod', fontsize=20)
ax4.tick_params(axis='y', labelsize=22)
ax1.tick_params(axis='x', labelsize=22)
ax3.tick_params(axis='x', labelsize=22)
ax1.set_ylim([75,90])
ax2.set_ylim([45,60])
ax3.set_ylim([75,90])
ax4.set_ylim([45,60])
ax1.grid(True, linestyle='--', alpha=0.5)
ax2.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper left', fontsize=22)
ax2.legend(loc='upper right',fontsize=22)
ax3.grid(True, linestyle='--', alpha=0.5)
ax4.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper left', fontsize=22)
ax4.legend(loc='upper right',fontsize=22)
fig.tight_layout()
# plt.savefig(os.path.join('parameter.pdf'), format='pdf')
plt.show()
