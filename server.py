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

# import numpy as np
# from av import VideoFrame
# import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing as mp

from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.signaling import TcpSocketSignaling, BYE

from media import MediaRelay
from model import SingleNetwork
from dataset import RecentBiasDataset
from misc import Patch, bytes_to_ndarray

logger = logging.getLogger('server')
relay = MediaRelay()  # a media source that relays one or more tracks to multiple consumers.


class TrackScheduler:
    """
    Track scheduler is used in server to schedule a track.

    - invoke set_track when track is ready
    - async call to get_track will return the internal track
    - async call to start_consuming will consume frames from the head of the track
    - invoke stop_consuming to stop consuming frames
    """

    def __init__(self):
        self._event = asyncio.Event()
        self._track = None
        self._signal = False

    def set_track(self, track):
        self._track = track
        self._event.set()

    async def get_track(self):
        await self._event.wait()
        return self._track

    async def start_consuming(self):
        await self._event.wait()
        while not self._signal:
            await self._track.recv()

    def stop_consuming(self):
        self._signal = True


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
        self.model.load_state_dict(torch.load('data/pretrained/epoch_20.pt'))  # TODO load pretrained before train

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
            self.__log_info('keyboard interrupt training')
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


async def comm_sender(pc, signaling, patch_queue, track_scheduler):
    """
    Communicates with sender
    """

    def log_info(msg, *args):
        logger.info(f'@Sender {msg}', *args)

    await signaling.connect()
    log_info('Signaling connected')

    pc.addTransceiver('video', direction='recvonly')
    pc.addTransceiver('audio', direction='recvonly')

    @pc.on('track')
    def on_track(track):
        """
        Callback function for receiving track from client
        """
        log_info(f'Received {track.kind} track')
        if track.kind == 'video':
            track_scheduler.set_track(relay.subscribe(track))  # track not works
            log_info('Got track from sender')
            asyncio.create_task(track_scheduler.start_consuming())  # start consuming frames
        else:
            # Not consider audio at this stage
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
        # cv2.imwrite(f'data/360p/{i:04d}_lr.png', lr_array)
        # cv2.imwrite(f'data/720p/{i:04d}_hr.png', hr_array)
        i += 1

    @patch_channel.on('close')
    def on_patch_channel_close():
        log_info('patch channel close')

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


async def comm_receiver(pc, signaling, track_scheduler):
    def log_info(msg):
        logger.info(f'@Receiver {msg}')

    await signaling.connect()
    log_info('Signaling connected')

    track = await track_scheduler.get_track()

    pc.addTrack(track)  # work
    # pc.addTrack(relay.subscribe(track))
    log_info('Set track for receiver')

    # # dummy channel
    # dummy_channel = pc.createDataChannel('dummy')
    #
    # @dummy_channel.on('open')
    # def on_dummy_channel_open():
    #     dummy_channel.send('I am server')
    #
    # @dummy_channel.on('message')
    # def on_dummy_channel_message(msg):
    #     # log_info(f'received {msg}')
    #     dummy_channel.send('I am server')
    #
    # @dummy_channel.on('close')
    # def on_dummy_channel_close():
    #     log_info('dummy channel close')

    await pc.setLocalDescription(await pc.createOffer())  # create SDP offer and set as local description
    await signaling.send(pc.localDescription)  # send local description to signal server

    # consume signaling
    while True:
        obj = await signaling.receive()

        if isinstance(obj, RTCSessionDescription):
            log_info('Received remote description')
            await pc.setRemoteDescription(obj)
            track_scheduler.stop_consuming()  # stop consuming frames
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
    # parser.add_argument('--record-dir', type=str, default='result/records', help='Directory for media records')
    # parser.add_argument('--record-sr-fn', type=str, default='sr.mp4', help='SR video record name')
    # parser.add_argument('--record-raw-fn', type=str, default='raw.mp4', help='Raw video record name')
    # parser.add_argument('--not-record-sr', action='store_true')
    # parser.add_argument('--not-record-raw', action='store_true')
    # parser.add_argument('--hr-height', type=int, default=720)
    # parser.add_argument('--lr-height', type=int, default=360)
    # parser.add_argument('--fps', type=int, default=5)

    # model
    parser.add_argument('--not-use-cuda', action='store_true')
    parser.add_argument('--model-scale', type=int, default=2)
    parser.add_argument('--model-num-blocks', type=int, default=6)
    parser.add_argument('--model-num-features', type=int, default=6)

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

    # signaling
    parser.add_argument('--signaling-host', type=str, default='127.0.0.1', help='TCP socket signaling host')  # 192.168.0.201
    parser.add_argument('--signaling-port-sender', type=int, default=9999, help='TCP socket signaling port for sender side')
    parser.add_argument('--signaling-port-receiver', type=int, default=10001, help='TCP socket signaling port for receiver side')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # logging settings
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(level=logging.DEBUG if args.debug else logging.INFO)

    # train at another process
    mp.set_start_method('spawn', force=True)
    patch_queue = mp.Queue()
    train_process = mp.Process(target=run_trainer, args=(patch_queue, args))
    train_process.start()

    # RTC
    sender_signaling = TcpSocketSignaling(args.signaling_host, args.signaling_port_sender)
    sender_pc = RTCPeerConnection()
    receiver_signaling = TcpSocketSignaling(args.signaling_host, args.signaling_port_receiver)
    receiver_pc = RTCPeerConnection()

    # track scheduler
    track_scheduler = TrackScheduler()

    # run server - connects sender and receiver
    loop = asyncio.get_event_loop()
    try:
        sender_coro = comm_sender(sender_pc, sender_signaling, patch_queue, track_scheduler)
        receiver_coro = comm_receiver(receiver_pc, receiver_signaling, track_scheduler)
        loop.run_until_complete(asyncio.gather(sender_coro, receiver_coro))

    except KeyboardInterrupt:
        logger.info('keyboard interrupt while running server')
    finally:
        # cleanup
        loop.run_until_complete(sender_signaling.close())
        loop.run_until_complete(receiver_signaling.close())
        logger.info('Signaling close')

        loop.run_until_complete(sender_pc.close())  # pc closes then no track
        loop.run_until_complete(receiver_pc.close())
        logger.info('pc close')

    patch_queue.close()
    train_process.terminate()
