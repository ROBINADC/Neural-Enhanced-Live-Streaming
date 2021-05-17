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
import asyncio

import numpy as np
from av import VideoFrame
import cv2
import torch

from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, RTCDataChannel, MediaStreamTrack
from aiortc.contrib.signaling import BYE, TcpSocketSignaling

from media import MediaRelay, MediaRecorderDelta, MediaBlackhole
from misc import get_resolution
from model import SingleNetwork

logger = logging.getLogger('Receiver')
relay = MediaRelay()  # a media source that relays one or more tracks to multiple consumers.


class DummyProcessor:
    def process(self, image: np.ndarray) -> np.ndarray:
        return image


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
        torch.set_grad_enabled(False)
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

    def __init__(self, track, processor):
        """
        :param track: the original track to be processed
        """
        super().__init__()
        self.track = track
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
        img = frame.to_ndarray(format='bgr24')
        img = self.processor.process(img)

        # rebuild a VideoFrame, preserving timing information
        new_frame = VideoFrame.from_ndarray(img, format='bgr24')
        new_frame.pts = frame.pts  # Presentation TimeStamps, denominated in terms of timebase, here
        new_frame.time_base = frame.time_base  # a unit of time, here Fraction(1, 90000) (of a second)

        return new_frame


async def comm_server(pc, signaling, processor, recorder_raw, recorder_sr):
    @pc.on('track')
    def on_track(track):
        logger.info('Received track from server')
        if track.kind == 'video':
            recorder_raw.addTrack(relay.subscribe(track))
            recorder_sr.addTrack(VideoProcessTrack(relay.subscribe(track), processor))
        else:
            # Not consider audio at this stage
            # recorder_raw.addTrack(track)  # add audio track to recorder_raw
            pass

    @pc.on('datachannel')
    def on_datachannel(channel: RTCDataChannel):
        logger.info('Received data channel: %s', channel.label)

        if channel.label == 'model':
            @channel.on('message')
            def on_message(msg):
                logger.info(f'receive msg')
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
            await recorder_sr.start()

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
    parser.add_argument('--process-type', type=str, default='sr', choices=('sr', 'grayish', 'none'))
    parser.add_argument('--debug', action='store_true', help='Set the logging verbosity to DEBUG')
    parser.add_argument('--log-dir', type=str, default='result/logs', help='Directory for logs')

    # video
    parser.add_argument('--record-dir', type=str, default='result/records', help='Directory for media records')
    parser.add_argument('--record-sr-fn', type=str, default='sr.mp4', help='SR video record name')
    parser.add_argument('--record-raw-fn', type=str, default='raw.mp4', help='Raw video record name')
    parser.add_argument('--not-record-sr', action='store_true')
    parser.add_argument('--not-record-raw', action='store_true')
    parser.add_argument('--hr-height', type=int, default=720, help='Height of origin high-resolution video')
    parser.add_argument('--lr-height', type=int, default=360, help='Height of transformed low-resolution video')
    parser.add_argument('--fps', type=int, default=5)

    # model
    parser.add_argument('--not-use-cuda', action='store_true')
    parser.add_argument('--model-scale', type=int, default=2)
    parser.add_argument('--model-num-blocks', type=int, default=6)
    parser.add_argument('--model-num-features', type=int, default=6)

    # inference
    parser.add_argument('--load-pretrained', action='store_true')
    parser.add_argument('--pretrained-fp', type=str)

    # signaling
    parser.add_argument('--signaling-host', type=str, default='127.0.0.1', help='TCP socket signaling host')  # 192.168.0.201
    parser.add_argument('--signaling-port', type=int, default=10001, help='TCP socket signaling port')
    args = parser.parse_args()

    os.makedirs(args.record_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger.setLevel(level=logging.DEBUG if args.debug else logging.INFO)

    # RTC
    signaling = TcpSocketSignaling(args.signaling_host, args.signaling_port)  # signaling server
    pc = RTCPeerConnection()

    # media sink
    low_resolution = get_resolution(args.lr_height)
    if args.record_dir and not args.not_record_raw:
        recorder_raw = MediaRecorderDelta(os.path.join(args.record_dir, args.record_raw_fn),
                                          logfile=os.path.join(args.log_dir, 'receiver_recorder_raw.log'),
                                          width=low_resolution.width, height=low_resolution.height, fps=args.fps)
    else:
        recorder_raw = MediaBlackhole()

    high_resolution = get_resolution(args.hr_height)
    if args.record_dir and not args.not_record_sr:
        recorder_sr = MediaRecorderDelta(os.path.join(args.record_dir, args.record_sr_fn),
                                         logfile=os.path.join(args.log_dir, 'receiver_recorder_sr.log'),
                                         width=high_resolution.width, height=high_resolution.height, fps=args.fps)
    else:
        recorder_sr = MediaBlackhole()

    # inference
    processor = None
    if args.process_type == 'sr':
        processor = SuperResolutionProcessor(args)
    elif args.process_type == 'none':
        processor = DummyProcessor()
    else:
        logger.info(f'Process type "{args.process_type}" is not recognized. Exit.')
        exit(0)

    # run receiver
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(comm_server(pc, signaling, processor, recorder_raw, recorder_sr))
    except KeyboardInterrupt:
        logger.info('keyboard interrupt while running receiver')
    finally:
        # cleanup
        loop.run_until_complete(recorder_raw.stop())  # work
        loop.run_until_complete(recorder_sr.stop())
        loop.run_until_complete(signaling.close())
        loop.run_until_complete(pc.close())  # pc closes then no track

