"""
This is used to
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

PATCH_SIZE = 21.36  # average patch size in kbits (2670 bytes)

pfs = [1, 3, 5, 7, 10, 20]  # patch frequency
bws = [pf * PATCH_SIZE for pf in pfs]  # bandwidth
qs = [23.49, 23.71, 24.24, 24.49, 24.80, 24.97]  # quality in PSNR

fig, ax1 = plt.subplots(figsize=(4, 3))

color = 'tab:red'
ax1.set_xlabel(r'Patch frequency ($\mathregular{s^{-1}}$)')
ax1.set_ylabel('Bandwidth (kbps)')
ax1.plot(pfs, bws, color=color, label='Bandwidth', linewidth=2)

color = 'tab:blue'
ax2 = ax1.twinx()
ax2.plot(pfs, qs, color=color, label='PSNR', linewidth=2)
ax2.scatter(pfs, qs, c=color, marker='+')
ax2.set_ylabel('PSNR (dB)')

plt.xticks(range(0, 21, 5))
ax1.set_yticks(range(0, 501, 100))
ax2.set_yticks(np.arange(23, 25.1, 0.5))

line_1, label_1 = ax1.get_legend_handles_labels()
line_2, label_2 = ax2.get_legend_handles_labels()

plt.legend(line_1 + line_2, label_1 + label_2, loc=4)
fig.tight_layout()
# plt.show()
plt.savefig('../result/quality/temp_bw_patch.svg', bbox_inches='tight', pad_inches=0.01)
