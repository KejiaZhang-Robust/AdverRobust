import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np

lamda = [0, 0.1,0.25,0.5,0.75,1.0]
lamda_clean = [80.58, 81.27, 80.89, 80.54, 80.11, 79.92]
lamda_PGD = [53.74, 54.69, 54.95, 55.07, 55.20, 55.31]

beta = [0, 0.1,0.25,0.5,0.75,1.0]
beta_clean = [80.18, 81.27, 81.44, 81.63, 81.70, 81.81]
beta_PGD = [51.92, 54.69, 54.45, 54.33, 54.20, 54.09]

fig, (ax1,ax3) = plt.subplots(1, 2, figsize=(18,9))
# ($\lambda$)
ax1.plot(lamda, lamda_clean, label='Clean',linewidth=3.5, marker='o', alpha=0.9,markersize=10, color='indianred')
ax1.set_xlabel('$\lambda$', fontsize=28)
ax1.set_ylabel('Accuracy(%)', fontsize=28)
ax1.tick_params(axis='y', labelsize=25, colors='indianred')
# ax1.set_title('FMR-GC at Conv.1', fontsize=28)

ax2 = ax1.twinx()
ax2.plot(lamda, lamda_PGD, label='PGD-10', linewidth=3.5, marker='s', alpha=0.9,markersize=10,color='teal')
ax2.set_ylabel('Robust Accuracy(%)', fontsize=28)
ax2.tick_params(axis='y', labelsize=25, colors='teal')

# ,color='seagreen',color='darkgoldenrod'
ax3.plot(beta, beta_clean, label='Clean', linewidth=3.5, marker='o', alpha=0.9,markersize=10, color='indianred')
ax3.set_xlabel(r'$\beta$', fontsize=28)
ax3.set_ylabel('Accuracy(%)', fontsize=28)
# ax3.set_title('FMR-GC at Conv.3', fontsize=28)
ax3.tick_params(axis='y', labelsize=25, colors='indianred')
# r'PGD-10 ($\beta$)'
ax4 = ax3.twinx()
ax4.plot(beta, beta_PGD, label='PGD-10', linewidth=3.5, marker='s', alpha=0.9,markersize=10, color='teal')
ax4.set_ylabel('Robust Accuracy(%)', fontsize=28)
ax4.tick_params(axis='y', labelsize=25,colors='teal')

ax1.tick_params(axis='x', labelsize=25)
ax3.tick_params(axis='x', labelsize=25)
ax1.set_ylim([79,85])
ax2.set_ylim([51,56])
ax3.set_ylim([79,85])
ax4.set_ylim([51,56])
ax1.grid(True, linestyle='--', alpha=0.5)
ax2.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper left', fontsize=25)
ax2.legend(loc='upper right',fontsize=25)
ax3.grid(True, linestyle='--', alpha=0.5)
ax4.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper left', fontsize=25)
ax4.legend(loc='upper right',fontsize=25)
fig.tight_layout()
plt.savefig(os.path.join('./plot/parameter.pdf'), format='pdf')
plt.show()
