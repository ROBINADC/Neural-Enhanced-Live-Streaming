"""
Real-time video streaming receiver
- receive raw video from server
- receive SR models from server
- applies per-frame super-resolution and displays (stores at this stage) the high-resolution video
"""

__author__ = "Yihang Wu"

from io import BytesIO
import os
from fractions import Fraction
import argparse
import logging
import asyncio
import queue

import numpy as np
from av import VideoFrame
import torch

from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, RTCDataChannel, MediaStreamTrack, RTCConfiguration
from aiortc.contrib.signaling import BYE, TcpSocketSignaling

from media import MediaRelay, MediaRecorderDelta, MediaBlackhole
from misc import ClassLogger, Resolution, get_ice_servers
from model import SingleNetwork

logger = logging.getLogger('receiver')
relay = MediaRelay()  # a media source that relays one or more tracks to multiple consumers.


class DummyProcessor:
    def process(self, image: np.ndarray) -> np.ndarray:
        return image


class SuperResolutionProcessor(ClassLogger):
    """
    Processor for super-resolving a frame.
    The SR model is replaced by the new model presented in model queue.
    """

    def __init__(self, args):
        super(SuperResolutionProcessor, self).__init__('receiver')

        self.model = SingleNetwork(args.model_scale, num_blocks=args.model_num_blocks,
                                   num_channels=3, num_features=args.model_num_features)
        self.load_pretrained = args.load_pretrained
        self.pretrained_fp = args.pretrained_fp
        self.device = 'cuda' if args.use_gpu else 'cpu'

        self._model_queue = queue.SimpleQueue()  # for incoming models

        self._setup()

    def _setup(self):
        if self.device == 'cuda':
            self.model = self.model.half().to(self.device)
        else:
            self.model = self.model.to(self.device)  # pytorch conv cpu version not support fp16

        # using pretrained model at receiver side is trivial (as it is replaced by a new model in short time)
        if self.load_pretrained and self.pretrained_fp and os.path.exists(self.pretrained_fp):
            self.model.load_state_dict(torch.load(self.pretrained_fp))
            self.log_info('load pretrained model (NOT RECOMMAND)')

        self.model.eval()
        torch.set_grad_enabled(False)
        self.log_info('finish setup')

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Super-resolve a frame represented in uint8 ndarray
        """

        # before processing a frame, check whether the model can be updated
        # current implementation checks at every frame, which can be refined later
        self._update_model()

        x = torch.from_numpy(image).byte().to(self.device)  # (lr_height, lr_width, 3), torch.uint8
        x = x.permute(2, 0, 1)  # (3, lr_height, lr_width)
        if self.device == 'cuda':
            x = x.half()  # equivalent to x.to(torch.float16)
        else:
            x = x.float()  # equivalent to x.to(torch.float32)

        x.div_(255)
        x.unsqueeze_(0)  # (1, 3, lr_height, lr_width)

        out = self.model(x)  # (1, 3, hr_height, hr_width)
        out = out.data[0].permute(1, 2, 0)  # (hr_height, hr_width, 3)
        out = out * 255
        out = torch.clamp(out, 0, 255)
        out = out.byte()  # transform back to torch.uint8

        hr_image = out.cpu().numpy()  # (hr_height, hr_width, 3)
        return hr_image

    def _update_model(self):
        """
        Update the model using the newest model in queue.
        This method is invoked in method process
        """
        if self._model_queue.qsize() == 0:
            return

        m = self._model_queue.get_nowait()
        while self._model_queue.qsize() > 0:  # get the newest model
            m = self._model_queue.get_nowait()
        self.model.load_state_dict(torch.load(BytesIO(m)))
        self.log_info('update model')

    @property
    def model_queue(self):
        """
        A SimpleQueue object for placing newly trained models.
        The model in the queue should be saved by torch.save(), and represented in bytes beforehand.
        Models are loaded by torch.load(BytesIO(m)) in this class.

        Returns: queue for models
        """
        return self._model_queue


class VideoProcessTrack(MediaStreamTrack):
    """
    A video stream track that processes frames from an another track
    """

    kind = 'video'

    def __init__(self, track, processor):
        """
        Args:
            track (): the original track to be processed
            processor (SuperResolutionProcessor):
        """
        super().__init__()
        self.track = track
        self.processor = processor

        self.count = 0

    async def recv(self):
        """
        Generate the next VideoFrame
        """
        frame = await self.track.recv()  # read next frame from the original track
        img = frame.to_ndarray(format='bgr24')
        img = self.processor.process(img)

        # rebuild a VideoFrame, preserving timing information
        new_frame = VideoFrame.from_ndarray(img, format='bgr24')
        new_frame.pts = frame.pts  # Presentation TimeStamps, denominated in terms of timebase, here
        new_frame.time_base = frame.time_base  # a unit of time, here Fraction(1, 90000) (of a second)

        return new_frame


async def comm_server(pc, signaling, processor, recorder_raw, recorder_sr):
    """
    Receiver communicates with server.
    It receives video and models from server.

    Args:
        pc (RTCPeerConnection): peer connection object
        signaling (TcpSocketSignaling): signaling proxy. Could be other signaling tool. See aiortc.contrib.signaling for more.
        processor (SuperResolutionProcessor): the processor used to conduct per-frame processing
        recorder_raw (MediaRecorderDelta): the recorder for the raw video
        recorder_sr (MediaRecorderDelta): the recorder for the super-resolved video
    """

    @pc.on('track')
    def on_track(track):
        logger.info('Received track from server')
        if track.kind == 'video':
            recorder_raw.addTrack(relay.subscribe(track))
            recorder_sr.addTrack(VideoProcessTrack(relay.subscribe(track), processor))
        else:
            # Not consider audio at this stage
            # recorder_raw.addTrack(track)
            pass

    @pc.on('datachannel')
    def on_datachannel(channel: RTCDataChannel):
        logger.info('Received data channel: %s', channel.label)

        if channel.label == 'model':
            if isinstance(processor, SuperResolutionProcessor):
                @channel.on('message')
                def on_message(msg):
                    processor.model_queue.put(msg)
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
    parser.add_argument('--not-sr', action='store_true', help='Not to perform per-frame super-resolution')
    parser.add_argument('--debug', action='store_true', help='Set the logging verbosity to DEBUG')

    # directory
    parser.add_argument('--log-dir', type=str, default='result/logs', help='Directory for logs')
    parser.add_argument('--record-dir', type=str, default='result/records', help='Directory for media records')

    # video
    parser.add_argument('--record-sr-fn', type=str, default='sr.mp4', help='SR video record name')
    parser.add_argument('--record-raw-fn', type=str, default='raw.mp4', help='Raw video record name')
    parser.add_argument('--not-record-sr', action='store_true', help='Do not record SR video')
    parser.add_argument('--not-record-raw', action='store_true', help='Do not record raw video')
    parser.add_argument('--aspect-ratio', type=str, default='4x3', help='Aspect ratio of the video given in "[W]x[H]"')
    parser.add_argument('--hr-height', type=int, default=480, help='Height of origin high-resolution video')
    parser.add_argument('--lr-height', type=int, default=240, help='Height of transformed low-resolution video')
    parser.add_argument('--fps', type=int, default=30)

    # model
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU to infer. Strongly recommand to use GPU for deep network')
    parser.add_argument('--model-scale', type=int, default=2)
    parser.add_argument('--model-num-blocks', type=int, default=8)
    parser.add_argument('--model-num-features', type=int, default=8)

    # inference
    parser.add_argument('--load-pretrained', action='store_true', help='Load pretrained model for super-resolution (NOT RECOMMAND)')
    parser.add_argument('--pretrained-fp', type=str, help='File path to the pretrained model')

    # signaling
    parser.add_argument('--signaling-host', type=str, default='127.0.0.1', help='TCP socket signaling host')  # 192.168.0.201
    parser.add_argument('--signaling-port', type=int, default=10001, help='TCP socket signaling port')

    # ICE server
    parser.add_argument('--ice-config', type=str, help='ICE server configuration')
    parser.add_argument('--ice-provider', type=str, default='google', help='ICE server provider')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.record_dir, exist_ok=True)

    # logging settings
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(level=logging.DEBUG if args.debug else logging.INFO)

    # RTC
    signaling = TcpSocketSignaling(args.signaling_host, args.signaling_port)

    if args.ice_config is None:
        logger.info('ice server is not configured')
        ice_servers = None
    else:
        logger.info(f'configure ice server from {args.ice_provider}')
        ice_servers = get_ice_servers(args.ice_config, args.ice_provider)  # a list of ice servers (might be empty)
    rtc_config = RTCConfiguration(iceServers=ice_servers)

    pc = RTCPeerConnection(configuration=rtc_config)

    aspect_ratio = Fraction(*map(int, args.aspect_ratio.split('x')))
    high_resolution = Resolution.get(args.hr_height, aspect_ratio)
    low_resolution = Resolution.get(args.lr_height, aspect_ratio)

    # media sink
    if args.record_dir and not args.not_record_raw:
        recorder_raw = MediaRecorderDelta(os.path.join(args.record_dir, args.record_raw_fn),
                                          logfile=os.path.join(args.log_dir, 'receiver_recorder_raw.log'),
                                          width=low_resolution.width, height=low_resolution.height, fps=args.fps)
    else:
        recorder_raw = MediaBlackhole()

    if args.record_dir and not args.not_record_sr:
        recorder_sr = MediaRecorderDelta(os.path.join(args.record_dir, args.record_sr_fn),
                                         logfile=os.path.join(args.log_dir, 'receiver_recorder_sr.log'),
                                         width=high_resolution.width, height=high_resolution.height, fps=args.fps)
    else:
        recorder_sr = MediaBlackhole()

    # SR processor
    if args.not_sr:
        processor = DummyProcessor()
    else:
        processor = SuperResolutionProcessor(args)

    # run receiver
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(comm_server(pc, signaling, processor, recorder_raw, recorder_sr))
    except KeyboardInterrupt:
        logger.info('keyboard interrupt while running receiver')
    finally:
        # cleanup
        loop.run_until_complete(recorder_raw.stop())
        loop.run_until_complete(recorder_sr.stop())
        loop.run_until_complete(signaling.close())
        loop.run_until_complete(pc.close())  # pc closes then no track
