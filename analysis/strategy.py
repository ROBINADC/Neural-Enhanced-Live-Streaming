"""
Plot PSNR gain over different optimization strategies.

Created on 2021/6/7 
"""

__author__ = "Yihang Wu"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_P1B1_CSV = '../result/archives/2.0_10_0.20_22.69_24.97.csv'
FILE_P1B0_CSV = '../result/archives/2.0_10_0.20_no-bias_22.69_24.72.csv'
FILE_P0B1_CSV = '../result/archives/2.0_10_0.20_no-psnr_22.69_24.09.csv'
FILE_P0B0_CSV = '../result/archives/2.0_10_0.20_no-bias_no-psnr_22.69_23.59.csv'
FILE_FIG = '../result/quality/temp_strategy.svg'

FPS = 5
SMOOTH_VALUE = 20  # group N data points and use their average

indices = None


def softing(a):
    return np.mean(np.reshape(a, (-1, SMOOTH_VALUE)), axis=1)


def get_psnr_diff(file):
    global indices

    df = pd.read_csv(file)

    xs = np.array(df['index'])
    if indices is None:
        indices = xs
    else:
        assert np.sum(indices - xs) == 0

    raws = np.array(df['raw'])
    srs = np.array(df['sr'])
    diff = srs - raws

    return diff


p1b1, p1b0, p0b1, p0b0 = map(get_psnr_diff, [FILE_P1B1_CSV, FILE_P1B0_CSV, FILE_P0B1_CSV, FILE_P0B0_CSV])
p1b1, p1b0, p0b1, p0b0 = map(softing, [p1b1, p1b0, p0b1, p0b0])

indices_smooth = indices[::SMOOTH_VALUE]
seconds_smooth = indices_smooth / FPS
mins_smooth = seconds_smooth / 60

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(mins_smooth, p1b1, label='PSNR Filter + Biased Sampling')
ax.plot(mins_smooth, p1b0, label='PSNR Filter only')
ax.plot(mins_smooth, p0b1, label='Biased Sampling only')
ax.plot(mins_smooth, p0b0, label='None of above applied')
plt.axhline(y=0, color='gray', alpha=0.3, ls='-', linewidth=0.5)

ax.xaxis.set_major_locator(plt.MultipleLocator(1))
# ax.xaxis.set_minor_locator(plt.MultipleLocator(20))

plt.xlabel('Time (min)')
plt.ylabel('PSNR Gain (dB)')

plt.legend()

# plt.show()
plt.savefig(FILE_FIG, bbox_inches='tight', pad_inches=0.02)
