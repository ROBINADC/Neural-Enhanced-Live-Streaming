"""
Plot bandwidth consumption to each model frequency and the corresponding PSNR.
Patch frequency is fixed in this case.

Created on 2021/6/6 
"""

__author__ = "Yihang Wu"

import numpy as np
import matplotlib.pyplot as plt

FILE_FIG = '../result/quality/temp_bw_model.svg'

MODEL_SIZE = 521.2  # average model size in kbits (estimated 65156 bytes)

mfs = [0.05, 0.1, 0.15, 0.175, 0.2]
bws = [mf * MODEL_SIZE for mf in mfs]
qs = [23.49, 24.24, 24.80, 24.89, 24.97]

fig, ax1 = plt.subplots(figsize=(4, 3))

color = 'tab:red'
ax1.set_xlabel(r'Model Frequency (Hz)')
ax1.set_ylabel('Model Bandwidth Usage (kbps)')
ax1.plot(mfs, bws, color=color, label='Bandwidth', linewidth=2)

color = 'tab:blue'
ax2 = ax1.twinx()
ax2.plot(mfs, qs, color=color, label='PSNR', linewidth=2)
ax2.scatter(mfs, qs, marker='+')
ax2.set_ylabel('PSNR (dB)')

plt.xticks(np.arange(0.05, 0.21, 0.05))
ax1.set_yticks(range(20, 110, 20))
ax2.set_yticks(np.arange(23, 25.1, 0.5))

line_1, label_1 = ax1.get_legend_handles_labels()
line_2, label_2 = ax2.get_legend_handles_labels()

plt.legend(line_1 + line_2, label_1 + label_2, loc=4)
fig.tight_layout()
# plt.show()
plt.savefig(FILE_FIG, bbox_inches='tight', pad_inches=0.02)
