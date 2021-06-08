"""
Plot surface for different combinations of (patch_freq, model_freq) and their average PSNR

Created on 2021/6/6 
"""

__author__ = "Yihang Wu"

import numpy as np
import matplotlib.pyplot as plt

pfs = [1, 5, 10, 20]  # patch frequency
mfs = [0.05, 0.1, 0.15, 0.2]  # model frequency

pfs_mesh, mfs_mesh = np.meshgrid(pfs, mfs)

qs = np.array([  # quality
    [22.21, 22.37, 22.72, 23.49],
    [23.82, 24.14, 24.15, 24.24],
    [24.14, 24.40, 24.57, 24.80],
    [24.53, 24.56, 24.62, 24.97]
])

bl_xs, bl_ys = np.meshgrid([1, 20], [0.05, 0.2])  # baseline
bl_qs = np.empty_like(bl_xs, dtype=np.float)
bl_qs.fill(22.69)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(pfs_mesh, mfs_mesh, qs,
                       color='cornflowerblue', alpha=0.5, edgecolors='royalblue', label='SR')
ax.scatter(pfs_mesh, mfs_mesh, qs, c='steelblue', s=8)
surf_baseline = ax.plot_surface(bl_xs, bl_ys, bl_qs,
                                color='salmon', alpha=0.3, edgecolors='coral', label='Raw')

ax.set_zlim(22.0, 25.0)
ax.zaxis.set_major_locator(plt.MultipleLocator(1))
ax.view_init(elev=18, azim=-136)

plt.xticks(pfs)
plt.yticks(mfs)

ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.zaxis.set_rotate_label(False)
ax.set_xlabel(r'Patch Frequency (Hz)', rotation=22)  # $\mathregular{s^{-1}}$
ax.set_ylabel(r'Model Frequency (Hz)', rotation=-21)
ax.set_zlabel('Average PSNR (dB)', rotation=90)

# for legend
surf_baseline._facecolors2d = surf_baseline._facecolor3d
surf_baseline._edgecolors2d = surf_baseline._edgecolor3d

surf._facecolors2d = surf._facecolor3d
surf._edgecolors2d = surf._edgecolor3d

ax.legend(loc='best')
# plt.show()
plt.savefig('../result/quality/temp_surface.svg', bbox_inches='tight', pad_inches=0.01)
