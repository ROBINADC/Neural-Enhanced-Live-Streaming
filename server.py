"""
Video conferencing server
"""

__author__ = "Yihang Wu"

import os
import argparse
import logging
import pickle
import threading
import time
import asyncio

import numpy as np
from av import VideoFrame
# import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing as mp

from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.signaling import TcpSocketSignaling, BYE

from media import MediaBlackhole, MediaRelay, MediaRecorderDelta
from model import SingleNetwork
from dataset import RecentBiasDataset
from misc import Patch, bytes_to_ndarray, get_resolution

logger = logging.getLogger('server')

track_recv_event = asyncio.Event()
relay = MediaRelay()  # a media source that relays one or more tracks to multiple consumers.
track_global = None


def run_trainer(patch_queue, args):
    """
    This function is run in another process.
    """
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)  # setup logging

    trainer = OnlineTrainer(patch_queue, args)
    trainer.run()


class OnlineTrainer:
    def __init__(self, patch_queue, args):
        self.patch_queue = patch_queue

        self.model = SingleNetwork(args.model_scale, num_blocks=args.model_num_blocks,
                                   num_channels=3, num_features=args.model_num_features)

        self.duration_per_epoch = args.duration_per_epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.device = 'cuda' if not args.not_use_cuda else 'cpu'

        self.dataset = RecentBiasDataset(num_items_per_epoch=args.num_items_per_epoch,
                                         num_biased_samples=args.num_biased_samples,
                                         bias_weight=args.bias_weight)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_func = self._get_loss_func(args.loss_type)

        self.save_ckpt = args.save_ckpt
        self.ckpt_dir = args.ckpt_dir

        self.logger = logging.getLogger('server')

        self.epoch = 0
        self.pending_patches = []

        self._setup()

    def _setup(self):
        self.model.to(self.device)
        # self.model.load_state_dict(torch.load('data/pretrained/epoch_10.pt'))  # TODO load pretrained before train

        self.__log_info('finish setup')

    def run(self):
        fetch_thread = threading.Thread(target=self._fetching_patch)
        fetch_thread.start()

        try:
            while not self.pending_patches:
                time.sleep(1)
        except KeyboardInterrupt:
            self.__log_info('interrupt while waiting the first batch of training patches')
            return

        self.__log_debug('got pending patches')

        try:
            while True:
                epoch_start_time = time.time()

                self.extend_dataset()
                self.train_one_epoch()
                self.save_model()

                if self.duration_per_epoch is not None:
                    elapse = time.time() - epoch_start_time
                    if elapse < self.duration_per_epoch:
                        self.__log_debug(f'current epoch duration {elapse:.2f}')
                        time.sleep(self.duration_per_epoch - elapse)
                    else:
                        self.__log_warning(f'current epoch duration {elapse:.2f} is greater than {self.duration_per_epoch}')
        except KeyboardInterrupt:
            self.__log_info('interrupt training')
        # fetch_thread.join()

    def extend_dataset(self):
        size = len(self.pending_patches)
        self.__log_debug(f'extend {size} patches')
        if size == 0:
            return

        self.dataset.extend(self.pending_patches[:size])
        del self.pending_patches[:size]

    def train_one_epoch(self):
        self.model.train()

        for iteration, (x, y) in enumerate(self.dataloader):
            time.sleep(0.01)  # ?? yield the gpu so that inference can use it
            x, y = x.to(self.device), y.to(self.device)  # (*, 3, patch_height, patch_width)

            self.optimizer.zero_grad()
            loss = self.loss_func(self.model(x), y)
            loss.backward()
            self.optimizer.step()

            if iteration % 10 == 0:
                self.__log_debug(f'{iteration} {loss.item()}')

        self.epoch += 1
        self.__log_info(f'finish training epoch {self.epoch}')

    def validate(self):
        pass

    def save_model(self):
        if self.save_ckpt:
            fn = os.path.join(self.ckpt_dir, f'epoch_{self.epoch}.pt')
            torch.save(self.model.state_dict(), fn)
            old_fn = os.path.join(self.ckpt_dir, f'epoch_{self.epoch - 5}.pt')
            if os.path.exists(old_fn):
                os.remove(old_fn)

    def _fetching_patch(self):
        while True:
            patch = self.patch_queue.get()  # (hr_patch, lr_patch)
            self.pending_patches.append((patch[1], patch[0]))  # (lr_patch, hr_patch)
        # while True:
        #     try:
        #         self.pending_patches.append(self.patch_queue.get())
        #     except (KeyboardInterrupt, SystemExit, EOFError) as exc:
        #         self.__log_info(exc)
        #         break

    def _get_loss_func(self, loss_type):
        if loss_type == 'l1':
            return nn.L1Loss()
        elif loss_type == 'l2':
            return nn.MSELoss()
        else:
            raise NotImplementedError

    def __log_warning(self, msg):
        self.logger.warning(f'[OnlineTrainer] {msg}')

    def __log_info(self, msg):
        self.logger.info(f'[OnlineTrainer] {msg}')

    def __log_debug(self, msg):
        self.logger.debug(f'[OnlineTrainer] {msg}')


async def comm_sender():
    """
    Communicates with streamer
    :return:
    :rtype:
    """

    pass


async def run_server(pc: RTCPeerConnection, signaling, process_type, processor, recorder_raw, recorder_sr,
                     patch_queue):
    await signaling.connect()
    logger.info('Signaling connected')

    pc.addTransceiver('video', direction='recvonly')
    pc.addTransceiver('audio', direction='recvonly')

    @pc.on('track')
    def on_track(track):
        """
        Callback function for receiving track from client
        """
        logger.info('Received %s track', track.kind)
        if track.kind == 'video':
            global track_global
            track_global = track
            track_recv_event.set()
            # relay.subscribe(track)

            # recorder_raw.addTrack(relay.subscribe(track))
            # recorder_sr.addTrack(VideoProcessTrack(relay.subscribe(track), process_type, processor))
        else:
            # Not consider audio at this stage
            # recorder_raw.addTrack(track)  # add audio track to recorder_raw
            pass

    # dummy channel
    dummy_channel = pc.createDataChannel('dummy')

    # patch channel
    patch_channel = pc.createDataChannel('patch')

    @patch_channel.on('open')
    def on_patch_channel_open():
        pass

    i = 0

    @patch_channel.on('message')
    def on_patch_channel_message(patch):
        nonlocal i
        patch: Patch = pickle.loads(patch)
        hr_bytes = patch.hr_patch
        lr_bytes = patch.lr_patch
        hr_array = bytes_to_ndarray(hr_bytes)
        lr_array = bytes_to_ndarray(lr_bytes)
        patch_queue.put((hr_array, lr_array))
        # cv2.imwrite(f'temp/{i:04d}_lr.png', lr_array)
        # cv2.imwrite(f'temp/{i:04d}_hr.png', hr_array)
        i += 1

    @patch_channel.on('close')
    def on_patch_channel_close():
        logger.info('patch channel close')

    await pc.setLocalDescription(await pc.createOffer())  # create SDP offer and set as local description
    await signaling.send(pc.localDescription)  # send local description to signal server

    # consume signaling
    while True:
        obj = await signaling.receive()

        if isinstance(obj, RTCSessionDescription):
            logger.info('Received remote description')
            await pc.setRemoteDescription(obj)
            # await recorder_raw.start()
            # await recorder_sr.start()
        elif isinstance(obj, RTCIceCandidate):
            logger.info('Received remote candidate')
            await pc.addIceCandidate(obj)
        elif obj is BYE:
            logger.info('Exiting')
            break


async def comm_receiver(pc, signaling):
    def log_info(msg):
        logger.info(f'@Receiver {msg}')

    await signaling.connect()
    log_info('Signaling connected')

    await track_recv_event.wait()
    log_info('got track')

    pc.addTrack(relay.subscribe(track_global))
    #
    # pc.addTransceiver('video', direction='recvonly')
    # pc.addTransceiver('audio', direction='recvonly')
    #
    # @pc.on('track')
    # def on_track(track):
    #     """
    #     Callback function for receiving track from client
    #     """
    #     logger.info('Received %s track', track.kind)
    #     if track.kind == 'video':
    #         recorder_raw.addTrack(relay.subscribe(track))
    #         recorder_sr.addTrack(VideoProcessTrack(relay.subscribe(track), process_type, processor))
    #     else:
    #         # Not consider audio at this stage
    #         # recorder_raw.addTrack(track)  # add audio track to recorder_raw
    #         pass

    # dummy channel
    dummy_channel = pc.createDataChannel('dummy')

    @dummy_channel.on('open')
    def on_dummy_channel_open():
        dummy_channel.send('I am server')

    @dummy_channel.on('message')
    def on_dummy_channel_message(msg):
        # log_info(f'received {msg}')
        dummy_channel.send('I am server')

    @dummy_channel.on('close')
    def on_dummy_channel_close():
        log_info('dummy channel close')

    await pc.setLocalDescription(await pc.createOffer())  # create SDP offer and set as local description
    await signaling.send(pc.localDescription)  # send local description to signal server

    # consume signaling
    while True:
        obj = await signaling.receive()

        if isinstance(obj, RTCSessionDescription):
            log_info('Received remote description')
            await pc.setRemoteDescription(obj)
        elif isinstance(obj, RTCIceCandidate):
            log_info('Received remote candidate')
            await pc.addIceCandidate(obj)
        elif obj is BYE:
            log_info('Exiting')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest server (offer peer)')
    parser.add_argument('--process-type', type=str, default='sr', choices=('sr', 'grayish', 'none'))
    parser.add_argument('--debug', action='store_true', help='Set the logging verbosity to DEBUG')
    parser.add_argument('--log-dir', type=str, default='result/logs', help='Directory for logs')

    # video
    parser.add_argument('--record-dir', type=str, default='result/records', help='Directory for media records')
    parser.add_argument('--record-sr-fn', type=str, default='sr.mp4', help='SR video record name')
    parser.add_argument('--record-raw-fn', type=str, default='raw.mp4', help='Raw video record name')
    parser.add_argument('--not-record-sr', action='store_true')
    parser.add_argument('--not-record-raw', action='store_true')
    parser.add_argument('--hr-height', type=int, default=1080)
    parser.add_argument('--lr-height', type=int, default=360)
    parser.add_argument('--fps', type=int, default=5)

    # model
    parser.add_argument('--not-use-cuda', action='store_true')
    parser.add_argument('--model-scale', type=int, default=3)
    parser.add_argument('--model-num-blocks', type=int, default=8)
    parser.add_argument('--model-num-features', type=int, default=8)

    # train
    parser.add_argument('--ckpt-dir', type=str, default='result/ckpt', help='Directory for training checkpoint')
    parser.add_argument('--save-ckpt', action='store_true', help='Save training checkpoints to local if set')
    parser.add_argument('--duration-per-epoch', type=int, default=5, help='The training thread will pause until the epoch takes certain duration')
    parser.add_argument('--num-items-per-epoch', type=int, default=3000)  # 3000
    parser.add_argument('--batch-size', type=int, default=64)  # 64
    parser.add_argument('--num-biased-samples', type=int, default=150)
    parser.add_argument('--bias-weight', type=int, default=4)
    parser.add_argument("--loss-type", type=str, default='l1', choices=('l1', 'l2'))
    parser.add_argument('--learning-rate', type=float, default=1e-4)

    # inference
    parser.add_argument('--load-pretrained', action='store_true')
    parser.add_argument('--pretrained-fp', type=str)

    # signaling
    parser.add_argument('--signaling-host', type=str, default='127.0.0.1', help='TCP socket signaling host')
    parser.add_argument('--signaling-port', type=int, default=9999, help='TCP socket signaling port')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.record_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # logging settings
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(level=logging.DEBUG if args.debug else logging.INFO)

    signaling = TcpSocketSignaling(args.signaling_host, args.signaling_port)  # signaling server
    pc = RTCPeerConnection()  # peer connection

    # media sink
    low_resolution = get_resolution(args.lr_height)
    if args.record_dir and not args.not_record_raw:
        recorder_raw = MediaRecorderDelta(os.path.join(args.record_dir, args.record_raw_fn),
                                          logfile=os.path.join(args.log_dir, 'server_recorder_raw.log'),
                                          width=low_resolution.width, height=low_resolution.height, fps=args.fps)
    else:
        recorder_raw = MediaBlackhole()

    high_resolution = get_resolution(args.hr_height)
    if args.record_dir and not args.not_record_sr:
        recorder_sr = MediaRecorderDelta(os.path.join(args.record_dir, args.record_sr_fn),
                                         logfile=os.path.join(args.log_dir, 'server_recorder_sr.log'),
                                         width=high_resolution.width, height=high_resolution.height, fps=args.fps)
    else:
        recorder_sr = MediaBlackhole()

    # train at another process
    mp.set_start_method('spawn', force=True)
    patch_queue = mp.Queue()
    train_process = mp.Process(target=run_trainer, args=(patch_queue, args))
    train_process.start()

    # inference
    # processor = SuperResolutionProcessor(args)
    processor = None

    # receiver
    pc_recv = RTCPeerConnection()
    signaling_recv = TcpSocketSignaling(args.signaling_host, 10001)

    # run server
    loop = asyncio.get_event_loop()
    try:
        # loop.run_until_complete(run_server(pc, signaling, args.process_type, processor, recorder_raw, recorder_sr, patch_queue))
        loop.run_until_complete(asyncio.gather(run_server(pc, signaling, args.process_type, processor, recorder_raw, recorder_sr, patch_queue),
                                               comm_receiver(pc_recv, signaling_recv)
                                               ))

    except KeyboardInterrupt:
        logger.info('keyboard interrupt while running server')
    finally:
        # cleanup
        loop.run_until_complete(signaling.close())
        logger.info('Signaling close')
        # loop.run_until_complete(recorder_raw.stop_after_finish())
        # loop.run_until_complete(recorder_sr.stop_after_finish())
        loop.run_until_complete(pc.close())  # pc closes then no track. though, recording can last long a little bit
        logger.info('pc close')

        loop.run_until_complete(signaling_recv.close())
        loop.run_until_complete(pc_recv.close())

    patch_queue.close()
    train_process.terminate()
