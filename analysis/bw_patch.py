"""
Plot bandwidth consumption to each patch frequency and the corresponding PSNR.
Model frequency is fixed in this case.

Patch information
- High-resolution portion: 40x40
- Low-resolution portion: 20x20
- jpg quality 95
- pickle

Created on 2021/6/6
"""

__author__ = "Yihang Wu"

import numpy as np
import matplotlib.pyplot as plt

FILE_FIG = '../result/quality/temp_bw_patch.svg'

PATCH_SIZE = 21.36  # average patch size in kbits (2670 bytes)

pfs = [1, 5, 7, 10, 15, 20]  # patch frequency
bws = [pf * PATCH_SIZE for pf in pfs]  # bandwidth
qs = [24.53, 24.56, 24.59, 24.62, 24.85, 24.97]  # quality in PSNR

fig, ax1 = plt.subplots(figsize=(4, 3))

color = 'tab:red'
ax1.set_xlabel(r'Patch Frequency (Hz)')
ax1.set_ylabel('Patch Bandwidth Usage (kbps)')
ax1.plot(pfs, bws, color=color, label='Bandwidth', linewidth=2)

color = 'tab:blue'
ax2 = ax1.twinx()
ax2.plot(pfs, qs, color=color, label='PSNR', linewidth=2)
ax2.scatter(pfs, qs, c=color, marker='+')
ax2.set_ylabel('PSNR (dB)')

plt.xticks(range(0, 21, 5))
ax1.set_yticks(range(0, 501, 100))
ax2.set_yticks(np.arange(24, 25.1, 0.5))

line_1, label_1 = ax1.get_legend_handles_labels()
line_2, label_2 = ax2.get_legend_handles_labels()

plt.legend(line_1 + line_2, label_1 + label_2, loc=4)
fig.tight_layout()
# plt.show()
plt.savefig(FILE_FIG, bbox_inches='tight', pad_inches=0.01)
