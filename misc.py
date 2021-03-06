"""
Miscellany.
Most are the functions or classes shared across different files.
"""

__author__ = "Yihang Wu"

import json
import math
from fractions import Fraction
import logging
import asyncio
from typing import List

import numpy as np
from av import VideoFrame
import cv2
import torch.nn as nn
from aiortc import RTCIceServer


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
    """
    A most-recent slot (a container with only one space) implemented in coroutine context.
    When producer adds an item to the slot, it empties the slot before adding the item.
    When consumer requests an item, it waits until there is one.
    """

    def __init__(self):
        self._queue = asyncio.Queue(maxsize=1)

    def put(self, obj):
        if not self._queue.empty():
            self._queue.get_nowait()
        self._queue.put_nowait(obj)

    async def get(self):
        return await self._queue.get()


class Patch:
    """
    Patch context
    The actual patch representation that is transmitted through WebRTC data channel.
    The fields timestamp and loc are not used for now.
    """

    def __init__(self, hr_patch=None, lr_patch=None, timestamp=None, loc=None):
        self.hr_patch = hr_patch
        self.lr_patch = lr_patch
        # self.timestamp = timestamp
        # self.loc = loc


class Resolution:
    """
    Resolution for 4:3 or 16:9 screen
    """
    ASPECT_RATIO_4_3 = Fraction(4, 3)
    ASPECT_RATIO_16_9 = Fraction(16, 9)

    height_to_width = {
        ASPECT_RATIO_4_3: {240: 320, 360: 480, 480: 640, 720: 960},
        ASPECT_RATIO_16_9: {270: 480, 360: 640, 540: 960, 720: 1280, 1080: 1920}
    }

    def __init__(self, _width, _height):
        self._width = _width
        self._height = _height

    @classmethod
    def get(cls, height: int, aspect_ratio: Fraction):
        try:
            width = Resolution.height_to_width[aspect_ratio][height]
        except KeyError as e:
            raise NotImplementedError(f'Invalid key {e}')
        return Resolution(width, height)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


"""
Frame related utilities
"""


def frame_to_ndarray(frame: VideoFrame) -> np.ndarray:
    return frame.to_ndarray(format='bgr24')  # (height, width, 3: BGR) np.uint8


def ndarray_to_bytes(a: np.ndarray) -> bytes:
    """
    Convert image in ndarray to bytes.
    The image is compressed in JPEG with quality level 95.
    """
    return cv2.imencode('.jpg', a, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()


def frame_to_bytes(frame: VideoFrame) -> bytes:
    ndarray = frame.to_ndarray(format='bgr24')  # (height, width, 3: BGR) np.uint8
    _bytes = cv2.imencode('.jpg', ndarray, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
    return _bytes


def bytes_to_ndarray(b: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)  #


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
    """
    Calculate peak signal-to-noise ratio (PSNR)
    """
    pred = pred.astype(np.int32)  # convert to int32 in case the array is uint8
    true = true.astype(np.int32)

    mse = np.mean((pred - true) ** 2)
    if mse == 0:
        return 100
    else:
        return 20 * math.log10(max_val / math.sqrt(mse))


def get_ice_servers(file: str = None, provider: str = None) -> List[RTCIceServer]:
    """
    Get a list of ICE servers that configures STUN / TURN servers from given json file,
    or an empty list if file is not provided.

    Args:
        file (): file that contains the server information
        provider (): the provider of the ICE servers

    Returns:
        A list of RTCIceServer objects.
    """
    if file is None:
        return list()

    with open(file, 'r') as fin:
        a = json.load(fin)

    if provider is None or provider not in a.keys():
        raise KeyError(f'Unrecognized provider: {provider}')

    server_list = a[provider]
    ice_servers = [RTCIceServer(**sv) for sv in server_list]

    return ice_servers


def count_model_parameters(model: nn.Module) -> int:
    """
    Get the number of trainable parameters in the neural network

    Args:
        model (nn.Module): neural network

    Returns:
        the number of trainable parameters in the neural network
    """

    count = 0
    for layer, param in model.named_parameters():
        if param.requires_grad:
            count += np.prod(param.shape).item()
        # print(layer, param.shape)
    # print(count)
    return count
