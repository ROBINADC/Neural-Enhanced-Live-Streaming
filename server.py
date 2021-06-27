"""
Real-time video streaming server
- relay video from sender to receiver
- receive training patches from sender
- train super-resolution model in another process
- deliver fresh models to receiver
"""

__author__ = "Yihang Wu"

from io import BytesIO
import os
import argparse
import logging
import pickle
import threading
import time
import asyncio

# import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing as mp

from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration
from aiortc.mediastreams import MediaStreamError
from aiortc.contrib.signaling import TcpSocketSignaling, BYE

from media import MediaRelay
from model import SingleNetwork
from dataset import RecentBiasDataset
from misc import ClassLogger, Patch, bytes_to_ndarray, get_ice_servers

logger = logging.getLogger('server')
relay = MediaRelay()  # a media source that relays one or more tracks to multiple consumers.


class ModelTransmitter(ClassLogger):
    """
    Send the model (represented in bytes) in the specified queue to given data channel.
    """

    def __init__(self, mp_model_queue: mp.Queue):
        super().__init__('server')

        self._mp_queue = mp_model_queue

        self._async_queue = asyncio.Queue()
        self._model_channel = None
        self._task = None

        threading.Thread(target=self._fetching_model).start()

    async def _run(self):
        """
        Continuously sending bytes model through data channel
        """
        while True:
            try:
                m: bytes = await self._async_queue.get()
                self._model_channel.send(m)
            except asyncio.CancelledError:
                self.log_info('cancel task for sending models')
                return

    def start(self):
        """
        Start transmitting bytes models through data channel
        """
        if not isinstance(self._model_channel, RTCDataChannel):
            raise TypeError
        self._task = asyncio.create_task(self._run())

    def stop(self):
        if self._task is not None:
            self._task.cancel()

    def _fetching_model(self):
        """
        Continuously fetching bytes model from multiprocessing queue to asyncio queue
        """
        while True:
            try:
                self._async_queue.put_nowait(self._mp_queue.get())
                self.log_debug(f'put model to async queue. Current queue size: {self._async_queue.qsize()}')
            except (KeyboardInterrupt, SystemExit, EOFError) as exc:
                self.log_info(f'thread(_fetching_model) terminates with exception {type(exc).__name__}')
                break

    @property
    def model_channel(self) -> RTCDataChannel:
        return self._model_channel

    @model_channel.setter
    def model_channel(self, channel: RTCDataChannel):
        self._model_channel = channel


class TrackScheduler(ClassLogger):
    """
    Track scheduler is used to schedule a track in the server.

    - invoke set_track when track is ready
    - async call to get_track will return the internal track
    - async call to start_consuming will consume frames from the head of the track
    - invoke stop_consuming to stop consuming frames
    """

    def __init__(self, logfile: str = None):
        super().__init__('server')

        self._event = asyncio.Event()
        self._track = None
        self._signal = False

        try:
            self._log = open(logfile, 'w')
        except (TypeError, FileNotFoundError):
            self._log = open(os.devnull, 'w')
        self._count = 0

    def set_track(self, track):
        self._track = track
        self._event.set()

    async def get_track(self):
        await self._event.wait()
        return self._track

    async def start_consuming(self):
        await self._event.wait()
        while not self._signal:
            try:
                await self._track.recv()
                self._count += 1
            except MediaStreamError:
                self.log_debug('stop consuming due to MediaStreamError')
                return
        else:
            self._log.write(f'{self._count}')
            self._log.close()

    def stop_consuming(self):
        self._signal = True


def run_trainer(patch_queue, model_queue, args):
    """
    Run the online trainer.
    This function is invoked in another process.
    """
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)  # setup logging

    trainer = OnlineTrainer(patch_queue, model_queue, args)
    trainer.run()


class OnlineTrainer(ClassLogger):
    """
    A class for online training.
    The trainer continuously fetches the training patches from the patch queue, and add them to the ExtendableDataset.
    The trainer trains the model using all patches in the dataset.
    After every training epoch, the model will be delivered to the model queue.
    """

    def __init__(self, patch_queue, model_queue, args):
        super().__init__('server')

        self.patch_queue = patch_queue
        self.model_queue = model_queue

        self.model = SingleNetwork(args.model_scale, num_blocks=args.model_num_blocks,
                                   num_channels=3, num_features=args.model_num_features)
        self.load_pretrained = args.load_pretrained
        self.pretrained_fp = args.pretrained_fp

        self.training_pattern = args.training_pattern  # choose from 'intermittent' and 'unceasing'

        self.duration_per_epoch = args.duration_per_epoch
        self.eps = 0.01  # (only used in unceasing training)
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.device = 'cuda' if args.use_gpu else 'cpu'

        self.dataset = RecentBiasDataset(num_items_per_epoch=args.num_items_per_epoch,
                                         num_biased_samples=args.num_biased_samples,
                                         bias_weight=args.bias_weight)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_func = self._get_loss_func(args.loss_type)

        self.save_ckpt = args.save_ckpt
        self.ckpt_dir = args.ckpt_dir

        self.epoch = 0
        self._epoch_start_time = None

        self._pending_patches = []

        self._setup()

    def _setup(self):
        self.model.to(self.device)
        if self.load_pretrained and self.pretrained_fp and os.path.exists(self.pretrained_fp):
            self.model.load_state_dict(torch.load(self.pretrained_fp))
            self.log_info('load pretrained model')
        self.model.train()

        self.log_info(f'training pattern has been set to {self.training_pattern}')
        self.log_info('finish setup')

    def run(self):
        fetch_thread = threading.Thread(target=self._fetching_patch)
        fetch_thread.start()

        try:
            while not self._pending_patches:
                time.sleep(1)
        except KeyboardInterrupt:
            self.log_info('interrupt while waiting the first batch of training patches')
            return

        self.log_debug('got pending patches')

        try:
            while True:
                self._epoch_start_time = time.time()  # update epoch start time

                if self.training_pattern == 'unceasing':
                    self._unceasing_train()
                else:
                    self._intermittent_train()
        except KeyboardInterrupt:
            self.log_info('keyboard interrupt training')

    def _intermittent_train(self):
        """
        Train one epoch of the model intermittently.
        If the epoch duration is shorter than the prescribed duration,
        the process wait until that much time.
        Then it starts next epoch.
        """

        self.model.train()
        self._extend_dataset()

        for iteration, (x, y) in enumerate(self.dataloader):
            x, y = x.to(self.device), y.to(self.device)  # (*, 3, patch_height, patch_width)

            self.optimizer.zero_grad()
            loss = self.loss_func(self.model(x), y)
            loss.backward()
            self.optimizer.step()

            if iteration % 10 == 0:
                self.log_debug(f'{iteration} {loss.item()}')

        self.epoch += 1
        self.log_info(f'finish training epoch {self.epoch}')

        self._on_model_ready()

        if self.duration_per_epoch is not None:
            elapse = time.time() - self._epoch_start_time
            if elapse < self.duration_per_epoch:
                self.log_debug(f'current epoch duration {elapse:.2f}')
                time.sleep(self.duration_per_epoch - elapse)
            else:
                self.log_warning(f'current epoch duration {elapse:.2f} is greater than {self.duration_per_epoch}')

    def _unceasing_train(self):
        """
        Train one epoch of the model unceasingly.
        The model keeps training until the time hits the prescribed epoch duration.
        We say such an iteration a main epoch,
        while the iteration indicated by the dataloader object is called a sub-epoch.

        This training pattern certainly outperforms the intermittent one.
        It is at the cost of computational resources.
        """
        assert self._epoch_start_time is not None and self.duration_per_epoch > 0

        self.model.train()
        while True:  # iteration indicated by dataloader
            self._extend_dataset()  # extend dataset on the start of every sub-epoch

            for iteration, (x, y) in enumerate(self.dataloader):  # iterate a sub-epoch
                if time.time() - self._epoch_start_time > self.duration_per_epoch - self.eps:
                    break
                x, y = x.to(self.device), y.to(self.device)  # (*, 3, patch_height, patch_width)

                self.optimizer.zero_grad()
                loss = self.loss_func(self.model(x), y)
                loss.backward()
                self.optimizer.step()

                if iteration % 10 == 0:
                    self.log_debug(f'{iteration} {loss.item()}')
            else:
                continue
            break

        self._on_model_ready()
        self.log_debug(f'current epoch duration {time.time() - self._epoch_start_time:.2f}')

    def _extend_dataset(self):
        size = len(self._pending_patches)
        self.log_debug(f'extend {size} patches')
        if size == 0:
            return

        self.dataset.extend(self._pending_patches[:size])
        del self._pending_patches[:size]

    def _on_model_ready(self):
        # convert the trained model to bytes and put it to model_queue
        buffer = BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        self.model_queue.put(buffer.read())  # model in bytes
        self.log_debug(f'put a bytes model to mp queue')

        # optionally save the model to local file
        if self.save_ckpt:
            fn = os.path.join(self.ckpt_dir, f'epoch_{self.epoch}.pt')
            torch.save(self.model.state_dict(), fn)
            old_fn = os.path.join(self.ckpt_dir, f'epoch_{self.epoch - 5}.pt')
            if os.path.exists(old_fn):
                os.remove(old_fn)

    def validate(self):
        pass

    def _fetching_patch(self):
        while True:
            patch = self.patch_queue.get()  # (hr_patch, lr_patch)
            self._pending_patches.append((patch[1], patch[0]))  # (lr_patch, hr_patch)
        # while True:
        #     try:
        #         self._pending_patches.append(self.patch_queue.get())
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


async def comm_sender(pc, signaling, patch_queue, track_scheduler):
    """
    Communicate with sender program.
    Receive track from sender and hand over it to the track scheduler.
    Receive patches from sender and add them to the patch queue.

    Args:
        pc (RTCPeerConnection): peer connection object
        signaling (TcpSocketSignaling): signaling proxy. Could be other signaling tool. See aiortc.contrib.signaling for more.
        patch_queue (mp.Queue): multiprocessing queue that is used to place the training patches (for trainer, as it runs in another process)
        track_scheduler (TrackScheduler): responsible to receive the media track sent from sender, and consume frames from head
    """

    def log_info(msg, *args):
        logger.info(f'@Sender: {msg}', *args)

    await signaling.connect()
    log_info('signaling connected')

    pc.addTransceiver('video', direction='recvonly')
    pc.addTransceiver('audio', direction='recvonly')

    @pc.on('track')
    def on_track(track):
        """
        Callback function for receiving track from client
        """
        log_info(f'received {track.kind} track')
        if track.kind == 'video':
            track_scheduler.set_track(relay.subscribe(track))  # track not works
            log_info('got track from sender')
            asyncio.create_task(track_scheduler.start_consuming())  # start consuming frames
        else:
            # Not consider audio at this stage
            pass

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
        # In case you need to generate offline training data ...
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
            log_info('received remote description')
            await pc.setRemoteDescription(obj)
        elif isinstance(obj, RTCIceCandidate):
            log_info('received remote candidate')
            await pc.addIceCandidate(obj)
        elif obj is BYE:
            log_info('exiting')
            break


async def comm_receiver(pc, signaling, model_queue, track_scheduler):
    """
    Communicate with receiver program.
    Send video to the receiver (obtain track from the track scheduler).
    Send SR models to the receiver using model transmitter.

    Args:
        pc (RTCPeerConnection): peer connection
        signaling (TcpSocketSignaling): signaling proxy. Could be other signaling tool. See aiortc.contrib.signaling for more.
        model_queue (mp.Queue): multiprocessing queue that is used to place SR models
        track_scheduler (TrackScheduler):
    """

    def log_info(msg):
        logger.info(f'@Receiver {msg}')

    await signaling.connect()
    log_info('signaling connected')

    track = await track_scheduler.get_track()
    pc.addTrack(track)
    log_info('Set track for receiver')

    # model channel
    model_channel = pc.createDataChannel('model')

    model_transmitter = ModelTransmitter(model_queue)
    model_transmitter.model_channel = model_channel

    @model_channel.on('open')
    def on_model_channel_open():
        log_info('model channel open')
        model_transmitter.start()

    @model_channel.on('close')
    def on_model_channel_close():
        model_transmitter.stop()
        log_info('model channel close')

    await pc.setLocalDescription(await pc.createOffer())
    await signaling.send(pc.localDescription)

    # consume signaling
    while True:
        obj = await signaling.receive()

        if isinstance(obj, RTCSessionDescription):
            log_info('received remote description')
            await pc.setRemoteDescription(obj)
            track_scheduler.stop_consuming()  # stop consuming frames
        elif isinstance(obj, RTCIceCandidate):
            log_info('received remote candidate')
            await pc.addIceCandidate(obj)
        elif obj is BYE:
            log_info('exiting')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time video streaming server')
    parser.add_argument('--debug', action='store_true', help='Set the logging verbosity to DEBUG')

    # directory
    parser.add_argument('--log-dir', type=str, default='result/logs', help='Directory for logs')
    parser.add_argument('--ckpt-dir', type=str, default='result/ckpt', help='Directory for training checkpoint')

    # model
    parser.add_argument('--model-scale', type=int, default=2)
    parser.add_argument('--model-num-blocks', type=int, default=8)
    parser.add_argument('--model-num-features', type=int, default=8)

    # train
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU to train. Default for using CPU.')
    parser.add_argument('--training-pattern', type=str, default='intermittent', choices=('intermittent', 'unceasing'))
    parser.add_argument('--save-ckpt', action='store_true', help='Save training checkpoints to local if set')
    parser.add_argument('--duration-per-epoch', type=float, default=5, help='The training thread will pause until the epoch takes certain duration')
    parser.add_argument('--num-items-per-epoch', type=int, default=3000, help='The number of training items per epoch')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-biased-samples', type=int, default=150)
    parser.add_argument('--bias-weight', type=int, default=4)
    parser.add_argument("--loss-type", type=str, default='l1', choices=('l1', 'l2'))
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--load-pretrained', action='store_true')
    parser.add_argument('--pretrained-fp', type=str)

    # signaling
    parser.add_argument('--signaling-host', type=str, default='127.0.0.1', help='TCP socket signaling host')  # 192.168.0.201
    parser.add_argument('--signaling-port-sender', type=int, default=9999, help='TCP socket signaling port for sender side')
    parser.add_argument('--signaling-port-receiver', type=int, default=10001, help='TCP socket signaling port for receiver side')

    # ICE server
    parser.add_argument('--ice-config', type=str, help='ICE server configuration (json file)')
    parser.add_argument('--ice-provider', type=str, default='google', help='ICE server provider')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # logging settings
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(level=logging.DEBUG if args.debug else logging.INFO)

    # RTC
    sender_signaling = TcpSocketSignaling(args.signaling_host, args.signaling_port_sender)
    receiver_signaling = TcpSocketSignaling(args.signaling_host, args.signaling_port_receiver)

    if args.ice_config is None:
        logger.info('ice server is not configured')
        ice_servers = None
    else:
        logger.info(f'configure ice server from {args.ice_provider}')
        ice_servers = get_ice_servers(args.ice_config, args.ice_provider)  # a list of ice servers (might be empty)
    rtc_config = RTCConfiguration(iceServers=ice_servers)

    sender_pc = RTCPeerConnection(configuration=rtc_config)
    receiver_pc = RTCPeerConnection(configuration=rtc_config)

    # train at another process
    mp.set_start_method('spawn', force=True)
    patch_queue = mp.Queue()
    model_queue = mp.Queue()
    train_process = mp.Process(target=run_trainer, args=(patch_queue, model_queue, args))
    train_process.start()

    # track scheduler
    track_scheduler = TrackScheduler(logfile=os.path.join(args.log_dir, 'server_consume_frame.log'))

    # run server - connects sender and receiver
    loop = asyncio.get_event_loop()
    try:
        sender_coro = comm_sender(sender_pc, sender_signaling, patch_queue, track_scheduler)
        receiver_coro = comm_receiver(receiver_pc, receiver_signaling, model_queue, track_scheduler)
        loop.run_until_complete(asyncio.gather(sender_coro, receiver_coro))
    except KeyboardInterrupt:
        logger.info('keyboard interrupt while running server')
    finally:
        # cleanup
        loop.run_until_complete(sender_signaling.close())
        loop.run_until_complete(receiver_signaling.close())
        logger.info('signaling close')

        loop.run_until_complete(sender_pc.close())  # pc closes then no track
        loop.run_until_complete(receiver_pc.close())
        logger.info('pc close')

    patch_queue.close()
    model_queue.close()
    train_process.terminate()
