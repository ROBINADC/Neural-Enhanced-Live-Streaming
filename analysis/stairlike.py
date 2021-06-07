"""
This is used to 
refer:
Created on 2021/6/6 
"""

__author__ = "Yihang Wu"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_CSV = '../result/archives/2.0_10_0.05_22.69_23.49.csv'
FILE_FIG = '../result/quality/temp_stairlike.svg'

FPS = 5
SMOOTH_VALUE = 5  # group N data points and use their average


def softing(a):
    return np.mean(np.reshape(a, (-1, SMOOTH_VALUE)), axis=1)


df = pd.read_csv(FILE_CSV)

indices = np.array(df['index'])
raws = np.array(df['raw'])
srs = np.array(df['sr'])
diff = srs - raws

indices_smooth = indices[::SMOOTH_VALUE]
seconds_smooth = indices_smooth / FPS
diff_smooth = softing(diff)

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
ax.plot(seconds_smooth, diff_smooth, label='SR quality minus raw quality', linewidth=2)

ax.xaxis.set_major_locator(plt.MultipleLocator(60))
ax.xaxis.set_minor_locator(plt.MultipleLocator(20))
plt.grid(axis='x', which='major', ls='--', linewidth=0.5)
plt.grid(axis='x', which='minor', ls='--', linewidth=0.5)

plt.xlabel('Time (s)')
plt.ylabel('PSNR Gain (dB)')

plt.legend()
# plt.show()
plt.savefig(FILE_FIG, bbox_inches='tight', pad_inches=0.01)
