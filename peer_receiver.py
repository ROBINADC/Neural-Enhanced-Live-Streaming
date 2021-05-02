"""
Video conferencing receiver peer.
Reciever has following functionality
- receives raw video from server
- applies per-frame super-resolution and displays (stores at this stage) the high-resolution video

However, the distinction of sender peer and receiver peer should be merged someday.
"""

__author__ = "Yihang Wu"

import os
import argparse
import logging
import random
import pickle
import platform
import asyncio

import numpy as np
from av import VideoFrame
import cv2
import torch


from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, RTCDataChannel, MediaStreamTrack
from aiortc.contrib.signaling import BYE, TcpSocketSignaling

from media import MediaPlayer, MediaRelay, MediaPlayerDelta, MediaRecorderDelta
from misc import Patch, MostRecentSlot, frame_to_ndarray, ndarray_to_bytes, cal_psnr, get_resolution
from model import SingleNetwork

logger = logging.getLogger('Receiver')
relay = MediaRelay()  # a media source that relays one or more tracks to multiple consumers.

class SuperResolutionProcessor:
    def __init__(self, args):
        self.model = SingleNetwork(args.model_scale, num_blocks=args.model_num_blocks,
                                   num_channels=3, num_features=args.model_num_features)
        self.load_pretrained = args.load_pretrained
        self.pretrained_fp = args.pretrained_fp
        self.device = 'cuda' if not args.not_use_cuda else 'cpu'

        self._setup()

    def _setup(self):
        self.model = self.model.half().to(self.device)
        if self.load_pretrained and os.path.exists(self.pretrained_fp):
            self.model.load_state_dict(torch.load(self.pretrained_fp))
            self.__log_info('load pretrained model')
        self.model.eval()
        torch.set_grad_enabled(False)  # affect other part ?
        self.__log_info('finish setup')

    def process(self, image: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(image).byte().to(self.device)  # (lr_height, lr_width, 3)
        x = x.permute(2, 0, 1).half()  # (3, lr_height, lr_width)
        x.div_(255)
        x.unsqueeze_(0)  # (1, 3, lr_height, lr_width)

        out = self.model(x)  # (1, 3, hr_height, hr_width)
        out = out.data[0].permute(1, 2, 0)
        out = out * 255
        out = torch.clamp(out, 0, 255)
        out = out.byte()

        hr_image = out.cpu().numpy()
        return hr_image

    def __log_info(self, msg):
        logger.info(f'[SuperResolutionProcessor] {msg}')


class VideoProcessTrack(MediaStreamTrack):
    """
    A video stream track that processes frames from an another track
    """

    kind = 'video'

    def __init__(self, track, process_type: str, processor=None):
        """
        :param track: the original track to be processed
        :param process_type:
        """
        super().__init__()
        self.track = track
        self.type = process_type
        self.processor = processor

        self.count = 0

        # self.timer = Timer()
        # self.timer.start()

    async def recv(self):
        """
        Receive (generate) the next VideoFrame
        :return: frame
        """
        frame = await self.track.recv()  # read next frame from origin track
        if self.type == 'sr':
            img = frame.to_ndarray(format='bgr24')
            img = self.processor.process(img)

            self.count += 1
            new_frame = VideoFrame.from_ndarray(img, format='bgr24')
            new_frame.pts = frame.pts  # Presentation TimeStamps, denominated in terms of timebase, here
            new_frame.time_base = frame.time_base  # a unit of time, here Fraction(1, 90000) (of a second)

            return new_frame
        elif self.type == 'grayish':
            img = frame.to_ndarray(format='bgr24')  # (frame.height, frame.width, 3)

            img[::2, ::2, :] = 128
            img[1::2, 1::2, :] = 128

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format='bgr24')
            new_frame.pts = frame.pts  # Presentation TimeStamps, denominated in terms of timebase, here
            new_frame.time_base = frame.time_base  # a unit of time, here Fraction(1, 90000) (of a second)
            # Multiply pts with time_base yields the playout time of a frame
            return new_frame
        elif self.type == 'none':
            return frame
        else:
            raise NotImplementedError



async def comm_server(pc, signaling):

    @pc.on('track')
    def on_track(track):
        logger.info('Received track from server')
        recorder_raw.addTrack(relay.subscribe(track))

    @pc.on('datachannel')
    def on_datachannel(channel: RTCDataChannel):
        logger.info('Received data channel: %s', channel.label)

        if channel.label == 'patch':
            # worker.patch_channel = channel
            # worker.start()
            pass
        elif channel.label == 'dummy':
            @channel.on('message')
            def on_message(msg):
                # logger.info(f'received {msg}')
                channel.send('I am receiver')
        else:
            raise NotImplementedError

    # connect signaling
    await signaling.connect()

    # consume signaling
    while True:
        try:
            obj = await signaling.receive()
        except ConnectionRefusedError:
            logger.info('Connection Refused by remote computer')
            logger.info('This may be becuase the signaling server has not been set up')
            break

        if isinstance(obj, RTCSessionDescription):
            logger.info('Received remote description')
            await pc.setRemoteDescription(obj)
            await recorder_raw.start()

            await pc.setLocalDescription(await pc.createAnswer())
            await signaling.send(pc.localDescription)
        elif isinstance(obj, RTCIceCandidate):
            logger.info('Received remote candidate')
            await pc.addIceCandidate(obj)
        elif obj is BYE:
            logger.info('Exiting')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conferencing peer (Receiver)')
    parser.add_argument('--debug', action='store_true', help='Set the logging verbosity to DEBUG')

    # video
    parser.add_argument('--hr-height', type=int, default=1080, help='Height of origin high-resolution video')
    parser.add_argument('--lr-height', type=int, default=360, help='Height of transformed low-resolution video')

    # signaling
    parser.add_argument('--signaling-host', type=str, default='127.0.0.1', help='TCP socket signaling host')  # 192.168.0.201
    parser.add_argument('--signaling-port', type=int, default=10001, help='TCP socket signaling port')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.setLevel(level=logging.DEBUG if args.debug else logging.INFO)

    signaling = TcpSocketSignaling(args.signaling_host, args.signaling_port)  # signaling server
    pc = RTCPeerConnection()

    recorder_raw = MediaRecorderDelta('data/video/aa.mp4',
                                      logfile=None,
                                      width=640,height=360,fps=5)

    # run receiver
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(comm_server(pc, signaling))
    except KeyboardInterrupt:
        logger.info('keyboard interrupt while running receiver')
    finally:
        # cleanup
        # loop.run_until_complete(recorder_raw.stop_after_finish())
        loop.run_until_complete(signaling.close())
        loop.run_until_complete(pc.close())
        loop.run_until_complete(recorder_raw.stop_after_finish())