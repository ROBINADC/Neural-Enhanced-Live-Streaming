"""
Miscellany
"""

__author__ = "Yihang Wu"

import re
import time
from datetime import timedelta
import math
import logging
from collections import namedtuple

from av import VideoFrame
import cv2
import numpy as np
import asyncio


class ClassLogger:
    def __init__(self, name):
        self._logger = logging.getLogger(name)
        self._cls = self.__class__.__name__

    def log_debug(self, msg, *args):
        self._logger.debug(f'[{self._cls}] {msg}', *args)

    def log_info(self, msg, *args):
        self._logger.info(f'[{self._cls}] {msg}', *args)

    def log_warning(self, msg, *args):
        self._logger.warning(f'[{self._cls}] {msg}', *args)


class MostRecentSlot:
    def __init__(self):
        self._queue = asyncio.Queue(maxsize=1)

    def put(self, obj):
        if not self._queue.empty():
            self._queue.get_nowait()
        self._queue.put_nowait(obj)

    async def get(self):
        return await self._queue.get()


class Patch:
    def __init__(self, hr_patch=None, lr_patch=None, timestamp=None, loc=None):
        self.hr_patch = hr_patch
        self.lr_patch = lr_patch
        self.timestamp = timestamp
        self.loc = loc


def frame_to_ndarray(frame: VideoFrame) -> np.ndarray:
    return frame.to_ndarray(format='bgr24')  # (height, width, 3: BGR) np.uint8


def ndarray_to_bytes(a: np.ndarray) -> bytes:
    return cv2.imencode('.jpg', a, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()


def frame_to_bytes(frame: VideoFrame) -> bytes:
    ndarray = frame.to_ndarray(format='bgr24')  # (height, width, 3: BGR) np.uint8
    _bytes = cv2.imencode('.jpg', ndarray, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
    return _bytes


def bytes_to_ndarray(b: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)


def frame_to_jpeg(frame: VideoFrame):
    """
    indicator
    """
    ndarray = frame.to_ndarray(format='bgr24')  # 'uint8'
    # ndarray = cv2.imread('g.png')
    cv2.imwrite('a.jpg', ndarray)
    enc_img = cv2.imencode('.jpg', ndarray, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()

    dec_img = cv2.imdecode(np.frombuffer(enc_img, dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite('b.jpg', dec_img)


def cal_psnr(pred, true, max_val):
    mse = np.mean((pred - true) ** 2)
    if mse == 0:
        return 100
    else:
        return 20 * math.log10(max_val / math.sqrt(mse))


Resolution = namedtuple('Resolution', ('width', 'height'))


def get_resolution(height_or_quality: int) -> Resolution:
    """
    Get resolution for 16:9 screen
    """
    if height_or_quality in (1080, 4):
        return Resolution(1920, 1080)
    elif height_or_quality in (720, 3):
        return Resolution(1280, 720)
    elif height_or_quality in (540, 2):
        return Resolution(960, 540)
    elif height_or_quality in (360, 1):
        return Resolution(640, 360)
    elif height_or_quality in (270, 0):
        return Resolution(480, 270)
    else:
        raise NotImplementedError


def atoi(text):
    return int(text) if text.isdigit() else text


def alphanum(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


class Timer:
    def __init__(self):
        self._start_time = None
        self._stop_time = None

    def start(self):
        self._start_time = time.time()

    def stop(self):
        self._stop_time = time.time()

    def tok(self):
        return timedelta(seconds=int(time.time() - self._start_time))

    def get_duration(self):
        return timedelta(seconds=int(self._stop_time - self._start_time))


def ffmpeg_run():
    """
    Handy function for using python bindings for FFmpeg

    References: https://kkroening.github.io/ffmpeg-python/
    """
    import ffmpeg
    ffmpeg.input('.mp4').trim(start=0, end=3).filter('fps', fps=5, round='up').output('out.mp4').run()


if __name__ == '__main__':
    pass
