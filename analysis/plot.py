"""
This is used to 
refer:
Created on 2021/6/5 
"""

__author__ = "Yihang Wu"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_CSV = '../result/quality/temp.csv'
SMOOTH_VALUE = 10  # group N data points and use their average

df = pd.read_csv(FILE_CSV)

indices = np.array(df['index'])
raws = np.array(df['raw'])
srs = np.array(df['sr'])

indices_smooth = indices[::SMOOTH_VALUE]
seconds_smooth = indices_smooth / SMOOTH_VALUE
softing = lambda a: np.mean(np.reshape(a, (-1, SMOOTH_VALUE)), axis=1)
raws_smooth = softing(raws)
srs_smooth = softing(srs)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.plot(seconds_smooth, raws_smooth, label='raw', linewidth=0.8)
ax.plot(seconds_smooth, srs_smooth, label='sr', linewidth=0.8)

ax.xaxis.set_major_locator(plt.MultipleLocator(60))
plt.xlabel('Seconds')
plt.ylabel('PSNR')

plt.legend()
# plt.show()
plt.savefig('../result/quality/temp.eps')
