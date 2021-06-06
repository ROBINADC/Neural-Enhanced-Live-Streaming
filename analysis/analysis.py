"""
Generate quantitative performance from videos (src, raw, sr)
Created on 2021/6/4 
"""

__author__ = "Yihang Wu"

import os
import math
from functools import partial

import numpy as np
import pandas as pd
import cv2

DIR_QUALITY = '../result/quality'
DIR_FRAMES = '../result/frames'

FILE_FRAME_CONSUME = '../result/logs/server_consume_frame.log'
FILE_VIDEO_SRC = '../data/video/dana_480p_5fps.mp4'
FILE_VIDEO_RAW = '../result/records/raw.mp4'
FILE_VIDEO_SR = '../result/records/sr.mp4'

BOOL_SAVE_FRAMES = False
INT_SKIP_UNTIL = 100
INT_END = 2400


def get_num_consumed():
    with open(FILE_FRAME_CONSUME, 'r') as fin:
        return int(str(fin.read()))


def gen_frames(file, *, start: int = 0, skip_until: int = 0, end: int = 0, save_frames: bool = False, tag: str = None):
    if save_frames and not isinstance(tag, str):
        raise TypeError('field "tag" is mandatory when "save_frames" is set')

    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        print('video is not valid')

    count = start

    while cap.isOpened():
        ret, frame = cap.read()
        if count >= skip_until:
            if not ret or count >= end:
                break
            yield frame
            if save_frames and count % 20 == 0:
                cv2.imwrite(os.path.join(DIR_FRAMES, f'{count:04d}_{tag}.png'), frame)
        count += 1

    cap.release()


def upsample(image, scale=2):
    new_shape = (image.shape[1] * scale, image.shape[0] * scale)
    return cv2.resize(image, dsize=new_shape, interpolation=cv2.INTER_CUBIC)


def cal_psnr(pred, true, max_val=255.):
    pred = pred.astype(np.int32)  # convert to int32 in case the array is uint8
    true = true.astype(np.int32)
    mse = np.mean((pred - true) ** 2)
    if mse == 0:
        return 100
    else:
        return 20 * math.log10(max_val / math.sqrt(mse))


if __name__ == '__main__':
    os.makedirs(DIR_QUALITY, exist_ok=True)
    os.makedirs(DIR_FRAMES, exist_ok=True)

    num_frames_consumed = get_num_consumed()  # should be correctly updated
    print(num_frames_consumed)

    gen_src_frames = partial(gen_frames, FILE_VIDEO_SRC, start=0, skip_until=INT_SKIP_UNTIL, end=INT_END,
                             save_frames=BOOL_SAVE_FRAMES, tag='src')
    gen_raw_frames = partial(gen_frames, FILE_VIDEO_RAW, start=num_frames_consumed, skip_until=INT_SKIP_UNTIL, end=INT_END,
                             save_frames=BOOL_SAVE_FRAMES, tag='raw')
    gen_sr_frames = partial(gen_frames, FILE_VIDEO_SR, start=num_frames_consumed, skip_until=INT_SKIP_UNTIL, end=INT_END,
                            save_frames=BOOL_SAVE_FRAMES, tag='sr')

    indices = []
    raw_psnrs = []
    sr_psnrs = []

    for i, src_frame, raw_frame, sr_frame in zip(range(INT_SKIP_UNTIL, INT_END), gen_src_frames(), gen_raw_frames(), gen_sr_frames()):
        indices.append(i)
        raw_psnrs.append(cal_psnr(upsample(raw_frame), src_frame))
        sr_psnrs.append(cal_psnr(sr_frame, src_frame))

    df = pd.DataFrame({'index': indices, 'raw': raw_psnrs, 'sr': sr_psnrs})
    df.to_csv(os.path.join(DIR_QUALITY, f'temp.csv'), index=False)

    print(np.mean(raw_psnrs))
    print(np.mean(sr_psnrs))

    # df = pd.read_csv('../result/quality/temp.csv')
    # print(np.mean(df['raw']))
    # print(np.mean(df['sr']))
