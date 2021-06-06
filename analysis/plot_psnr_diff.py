"""
This is used to 
refer:
Created on 2021/6/6 
"""

__author__ = "Yihang Wu"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_CSV = '../result/quality/temp.csv'
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

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.plot(seconds_smooth, diff_smooth, label='sr - raw', linewidth=1)

ax.xaxis.set_major_locator(plt.MultipleLocator(60))
plt.xlabel('Seconds')
plt.ylabel('PSNR Difference')

plt.legend()
# plt.show()
plt.savefig('../result/quality/temp_diff.eps')
