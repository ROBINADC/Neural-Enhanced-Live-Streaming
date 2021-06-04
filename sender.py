"""
Video conferencing sender peer
However, the distinction of sender peer and receiver peer should be merged someday.
"""

__author__ = "Yihang Wu"

import argparse
import logging
import random
from fractions import Fraction
import pickle
import platform
import asyncio

import numpy as np
# from av import VideoFrame
import cv2

from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration
from aiortc.contrib.signaling import BYE, TcpSocketSignaling

from media import MediaRelay, MediaPlayerDelta
from misc import ClassLogger, Resolution, Patch, MostRecentSlot, get_ice_servers, frame_to_ndarray, ndarray_to_bytes, cal_psnr

logger = logging.getLogger('sender')
relay = MediaRelay()  # a media source that relays one or more tracks to multiple consumers.


class PatchSampler:
    def __init__(self, hr_width, hr_height, lr_width, lr_height, patch_grid_height, patch_grid_width, *,
                 psnr_filter=True, strict_quantity=False):
        """
        A patch sampler to sample training patches from given frames.
        Specifically, given a pair of high-resolution (hr) frame and low-resolution (lr) frame,
        the sampler cuts both the hr and lr frame into small patches in terms of given grid,
        and select the patches at the same position.
        The sampler allows to sample multiple patches from a frame.

        Examples:
            An example initialization could be PatchSampler(1920, 1080, 640, 360, 16, 9)

        Args:
            hr_width (int): the width of the high-resolution image
            hr_height (int): the height of the high-resolution image
            lr_width (int): the width of the low-resolution image
            lr_height (int): the height of the low-resolution image
            patch_grid_height (int): the height of the patch grid
            patch_grid_width (int): the width of the patch grid
            psnr_filter (bool): only accept the patch with PSNR lower than the average of the frame
            strict_quantity (bool): force to sample at least a specified number of patches from a single frame
        """
        self.hr_image_width = hr_width
        self.hr_image_height = hr_height
        self.lr_image_width = lr_width
        self.lr_image_height = lr_height
        self.psnr_filter = psnr_filter
        self.strict_quantity = strict_quantity

        if self.strict_quantity:
            raise NotImplementedError

        self.hr_patch_width = hr_width // patch_grid_width
        self.hr_patch_height = hr_height // patch_grid_height
        self.lr_patch_width = lr_width // patch_grid_width
        self.lr_patch_height = lr_height // patch_grid_height
        self.num_patch_grids = patch_grid_height * patch_grid_width
        self.sampling_points = [(h, w) for h in range(patch_grid_height) for w in range(patch_grid_width)]

        self._hr_image = None
        self._lr_image = None
        self._global_psnr = None

    def place(self, hr_image: np.ndarray, lr_image: np.ndarray) -> None:
        """
        Place a pair of hr image and lr image into the sampler.
        Later samples will be drawn from these images.

        Args:
            hr_image (): hr image with shape (hr_height, hr_width, *)
            lr_image (): lr image with shape (lr_height, lr_width, *)
        """
        self._hr_image = hr_image
        self._lr_image = lr_image
        lr_image_f32 = self._lr_image.astype(np.float32)
        if self.psnr_filter:
            ip_image = cv2.resize(lr_image_f32, (self.hr_image_width, self.hr_image_height), interpolation=cv2.INTER_LINEAR)
            self._global_psnr = cal_psnr(ip_image, self._hr_image, max_val=255)

    def sample(self, n: int = 1, max_inspect: int = None) -> list:
        """
        Sample a number of patches from placed images.

        Args:
            n (int): the number of patches to sample
            max_inspect (int): the maximum number of patches inpsected.
                If set to None, then the sampler will probably inspect every location to get the required amount of patches.

        Returns:
            A list of (hr_patch, lr_patch)
        """
        # assert not self.strict_quantity or n <= max_inspect

        if max_inspect is None:
            max_inspect = self.num_patch_grids

        samples = []
        for h, w in random.sample(self.sampling_points, max_inspect):
            hr_patch = self._hr_image[
                       h * self.hr_patch_height: (h + 1) * self.hr_patch_height,
                       w * self.hr_patch_width: (w + 1) * self.hr_patch_width, :]
            lr_patch = self._lr_image[
                       h * self.lr_patch_height: (h + 1) * self.lr_patch_height,
                       w * self.lr_patch_width: (w + 1) * self.lr_patch_width, :]
            if self.psnr_filter:
                lr_patch_f32 = lr_patch.astype(np.float32)
                ip_patch = cv2.resize(lr_patch_f32, (self.hr_patch_width, self.hr_patch_height), interpolation=cv2.INTER_LINEAR)
                psnr = cal_psnr(ip_patch, hr_patch, max_val=255)
                if psnr >= self._global_psnr:
                    continue
            samples.append((hr_patch, lr_patch))
            if len(samples) == n:
                break

        return samples


class PatchTransmitter(ClassLogger):
    def __init__(self, patch_sampler: PatchSampler):
        """
        PatchTransmitter at sender side.
        The PatchTransmitter has following functionality:
        - sample training patches using the provided PatchSampler object
        - deliver training patch to RTC's data channel

        Args:
            patch_sampler (): patch sampler instance
        """
        super().__init__('sender')

        self._sampler = patch_sampler

        self._slot = MostRecentSlot()  # a wrap to a queue object that is passed to the MediaPlayerDelta, storing the "most recent" pair of frames
        self._patch_channel = None  # the RTC patch channel
        self._task = None

    async def _run(self):
        while True:
            try:
                await asyncio.sleep(0.5)  # sample frame every one second (can further adjust to listen some signal)
                hr_frame, lr_frame = await self._slot.get()  # this is not the most recent frame, but the frame head of recent frame 0~0.1s
                self._sampler.place(frame_to_ndarray(hr_frame),  # (hr_height, hr_width, 3)
                                    frame_to_ndarray(lr_frame))  # (lr_height, lr_width, 3)
                samples = self._sampler.sample(10)
                for hr_patch, lr_patch in samples:
                    hr_bytes = ndarray_to_bytes(hr_patch)
                    lr_bytes = ndarray_to_bytes(lr_patch)
                    patch = Patch(hr_bytes, lr_bytes)
                    patch_bytes = pickle.dumps(patch)
                    self._patch_channel.send(patch_bytes)
                self.log_debug(f'put {len(samples)} samples to patch channel')

            except asyncio.CancelledError:
                return

    def start(self):
        if not isinstance(self._patch_channel, RTCDataChannel):
            raise TypeError
        self._task = asyncio.create_task(self._run())

    def stop(self):
        if self._task is not None:
            self._task.cancel()

    @property
    def slot(self):
        """
        The slot that placed the most recent frame.
        """
        return self._slot

    @property
    def patch_channel(self) -> RTCDataChannel:
        return self._patch_channel

    @patch_channel.setter
    def patch_channel(self, channel: RTCDataChannel):
        self._patch_channel = channel


async def run_sender(pc: RTCPeerConnection, signaling, audio, video, patch_transmitter: PatchTransmitter):
    def add_senders():
        for t in pc.getTransceivers():
            if t.kind == 'audio' and audio:
                pc.addTrack(audio)
            elif t.kind == 'video' and video:
                pc.addTrack(video)

    @pc.on('datachannel')
    def on_datachannel(channel: RTCDataChannel):
        logger.info('Received data channel: %s', channel.label)

        if channel.label == 'patch':
            patch_transmitter.patch_channel = channel
            patch_transmitter.start()
        elif channel.label == 'dummy':
            pass
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

            add_senders()
            await pc.setLocalDescription(await pc.createAnswer())
            await signaling.send(pc.localDescription)
        elif isinstance(obj, RTCIceCandidate):
            logger.info('Received remote candidate')
            await pc.addIceCandidate(obj)
        elif obj is BYE:
            logger.info('Exiting')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conferencing peer (Sender)')
    parser.add_argument('--play-from', type=str, help='Read the media from a file and sent it.')
    parser.add_argument('--debug', action='store_true', help='Set the logging verbosity to DEBUG')

    # camera
    parser.add_argument('--use-camera', action='store_true', help='Use camera (--play-from is ignored if set)')
    parser.add_argument('--cam-framerate', type=str, default='30', help='Camera ingest frame rate')
    parser.add_argument('--cam-videosize', type=str, default='640x480', help='Camera ingest resolution')

    # video
    parser.add_argument('--framerate-degradation', type=int, default=1, help='Use only 1 frame every specified frames')
    parser.add_argument('--aspect-ratio', type=str, default='4x3', help='Aspect ratio of the video given in "[W]x[H]"')
    parser.add_argument('--hr-height', type=int, default=480, help='Height of origin high-resolution video')
    parser.add_argument('--lr-height', type=int, default=240, help='Height of transformed low-resolution video')
    parser.add_argument('--patch-grid-height', type=int, default=12, help='Height of the patch grid')
    parser.add_argument('--patch-grid-width', type=int, default=16, help='Width of the patch grid')

    # signaling
    parser.add_argument('--signaling-host', type=str, default='127.0.0.1', help='TCP socket signaling host')  # 192.168.0.201
    parser.add_argument('--signaling-port', type=int, default=9999, help='TCP socket signaling port')

    # ICE server
    parser.add_argument('--ice-config', type=str, help='ICE server configuration')
    parser.add_argument('--ice-provider', type=str, default='google', help='ICE server provider')
    args = parser.parse_args()

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

    pc = RTCPeerConnection(rtc_config)

    aspect_ratio = Fraction(*map(int, args.aspect_ratio.split('x')))
    high_resolution = Resolution.get(args.hr_height, aspect_ratio)
    low_resolution = Resolution.get(args.lr_height, aspect_ratio)

    patch_sampler = PatchSampler(hr_width=high_resolution.width, hr_height=high_resolution.height,
                                 lr_width=low_resolution.width, lr_height=low_resolution.height,
                                 patch_grid_height=args.patch_grid_height, patch_grid_width=args.patch_grid_width,
                                 psnr_filter=True)
    patch_transmitter = PatchTransmitter(patch_sampler)  # training patch worker

    # framerate degradation
    """
    When the capability of the receiving device is insufficient,
    degrade the framerate of the original stream with level FR_DEG.
    Sample 1 frame every FR_DEG frames.
    For example, when the origin video is 30 fps,
    setting FR_DEG to 3 degrades the video to 10 fps;
    setting FR_DEG to 6 degrades the video to 5 fps.
    """
    fr_deg = args.framerate_degradation
    logger.info(f'Framerate degradation level is set to {fr_deg}')

    # media source
    audio_track = None
    video_track = None
    if args.use_camera:
        # camera options (not all options are supported)
        options = {'framerate': args.cam_framerate, 'video_size': args.cam_videosize}
        if platform.system() == 'Linux':
            webcam = MediaPlayerDelta('/dev/video0', frame_width=low_resolution.width, frame_height=low_resolution.height,
                                      slot=patch_transmitter.slot, framerate_degradation=fr_deg, format='v4l2', options=options)
        else:
            raise NotImplementedError
        # audio_track = None
        video_track = relay.subscribe(webcam.video)
    else:  # play from local file
        if args.play_from is None:
            logger.info('need to specify local media file. Exit.')
            exit(0)
        player = MediaPlayerDelta(args.play_from, frame_width=low_resolution.width, frame_height=low_resolution.height,
                                  slot=patch_transmitter.slot, framerate_degradation=fr_deg)
        # audio_track = player.audio
        video_track = player.video

    # run sender
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_sender(pc, signaling, audio_track, video_track, patch_transmitter))
    except KeyboardInterrupt:
        logger.info('keyboard interrupt while running sender')
    finally:
        # cleanup
        patch_transmitter.stop()
        loop.run_until_complete(signaling.close())
        loop.run_until_complete(pc.close())
