"""
This is used to 
refer:
Created on 2021/6/5 
"""

__author__ = "Yihang Wu"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_CSV = '../result/archives/2.0_10_0.20_22.69_24.97.csv'
FILE_FIG = '../result/quality/temp_diff.svg'

FPS = 5
SMOOTH_VALUE = 10  # group N data points and use their average


def softing(a):
    return np.mean(np.reshape(a, (-1, SMOOTH_VALUE)), axis=1)


df = pd.read_csv(FILE_CSV)

indices = np.array(df['index'])
raws = np.array(df['raw'])
srs = np.array(df['sr'])

indices_smooth = indices[::SMOOTH_VALUE]
seconds_smooth = indices_smooth / FPS
mins_smooth = seconds_smooth / 60

raws_smooth = softing(raws)
srs_smooth = softing(srs)

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
ax.plot(mins_smooth, srs_smooth, label='SR', linewidth=1)
ax.plot(mins_smooth, raws_smooth, label='Raw', linewidth=1)

ax.xaxis.set_major_locator(plt.MultipleLocator(1))
plt.xlabel('Time (min)')
plt.ylabel('PSNR (dB)')

plt.legend()
# plt.show()
plt.savefig(FILE_FIG, pad_inches=0.0)
