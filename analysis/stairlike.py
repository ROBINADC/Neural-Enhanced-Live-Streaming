"""
This is used to 
refer:
Created on 2021/6/6 
"""

__author__ = "Yihang Wu"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_CSV = '../result/archives/2_10_0.05_22.69_23.49.csv'
SMOOTH_VALUE = 5  # group N data points and use their average

df = pd.read_csv(FILE_CSV)

indices = np.array(df['index'])
raws = np.array(df['raw'])
srs = np.array(df['sr'])
diff = srs - raws

indices_smooth = indices[::SMOOTH_VALUE]
seconds_smooth = indices_smooth / SMOOTH_VALUE
softing = lambda a: np.mean(np.reshape(a, (-1, SMOOTH_VALUE)), axis=1)
diff_smooth = softing(diff)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
ax.plot(seconds_smooth, diff_smooth, label='SR quality minus raw quality', linewidth=2)

ax.xaxis.set_major_locator(plt.MultipleLocator(60))
ax.xaxis.set_minor_locator(plt.MultipleLocator(20))
plt.grid(axis='x', which='major', ls='--', linewidth=0.5)
plt.grid(axis='x', which='minor', ls='--', linewidth=0.5)

plt.xlabel('Time (sec)')
plt.ylabel('PSNR Gain (dB)')

plt.legend()
# plt.show()
plt.savefig('../result/quality/temp_stairlike.eps', bbox_inches='tight', pad_inches=0.01)
