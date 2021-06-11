"""
An offline training example that uses SingleNetwork and FixedDataset.
The SingleNetwork uses training patches to train a super-resolution model.
Each patch consists of a low-resolution portion (40, 40) and a corresponding high-resolution one (80, 80).
If training data haven't been prepared, use FakeDataset to see what is going on.
"""

__author__ = "Yihang Wu"

import re
import argparse
import os
import glob

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, ToTensor

from model import SingleNetwork


class FixedDataset(Dataset):
    transform = Compose([ToTensor(), ])

    def __init__(self, lr_dir, hr_dir):
        super().__init__()

        self.lr_dir = lr_dir
        self.hr_dir = hr_dir

        self.lr_filenames = []
        self.hr_filenames = []

        self._setup()

    def _setup(self):
        self.lr_filenames.extend(glob.glob(f'{self.lr_dir}/*.png'))
        self.lr_filenames.sort(key=alphanum)

        self.hr_filenames.extend(glob.glob(f'{self.hr_dir}/*.png'))
        self.hr_filenames.sort(key=alphanum)

        self.lr_patches = [cv2.imread(fn) for fn in self.lr_filenames]
        self.hr_patches = [cv2.imread(fn) for fn in self.hr_filenames]

        assert len(self.lr_patches) == len(self.hr_patches)

    def __len__(self):
        return len(self.lr_filenames)

    def __getitem__(self, item):
        x = self.lr_patches[item]
        y = self.hr_patches[item]

        x_tensor = self.transform(x)
        y_tensor = self.transform(y)

        return x_tensor, y_tensor


def atoi(text):
    return int(text) if text.isdigit() else text


def alphanum(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


class FakeDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.size = 1000
        self.xs = torch.rand((self.size, 3, 40, 40))  # low-resolution patches
        self.ys = self.xs.repeat(1, 1, 2, 2)  # high-resolution patches
        self.ys.div_(2)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.xs[item], self.ys[item]


class Trainer:
    def __init__(self, model, dataset, ckpt_dir, batch_size, learning_rate, device):
        self.model = model
        self.dataset = dataset
        self.ckpt_dir = ckpt_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.loss_func = nn.L1Loss()

        self.epoch = 0

        self._setup()

    def _setup(self):
        self.model = self.model.to(self.device)

    def train_one_epoch(self):
        self.model.train()

        for iteration, (x, y) in enumerate(self.dataloader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            loss = self.loss_func(self.model(x), y)
            loss.backward()
            self.optimizer.step()

            if iteration % 10 == 0:
                print(f'{iteration} {loss.item()}')

        self.epoch += 1

    def validate(self):
        pass

    def save_model(self):
        save_path = os.path.join(self.ckpt_dir, f'epoch_{self.epoch}.pt')
        torch.save(self.model.state_dict(), save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Offline Training Example')
    parser.add_argument('--use-fake-dataset', action='store_true', help='Use fake dataset in case there are no data')

    parser.add_argument('--lr-dir', type=str, default='data/360p', help='Directory for low-resolution patches')
    parser.add_argument('--hr-dir', type=str, default='data/720p', help='Directory for high-resolution patches')
    parser.add_argument('--ckpt-dir', type=str, default='data/pretrained', help='Directory for checkpoint')

    parser.add_argument('--model-scale', type=int, default=2)
    parser.add_argument('--model-num-blocks', type=int, default=8)
    parser.add_argument('--model-num-features', type=int, default=8)

    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--use-gpu', action='store_true')

    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = 'cuda' if args.use_gpu else 'cpu'

    model = SingleNetwork(args.model_scale, num_blocks=args.model_num_blocks, num_channels=3, num_features=args.model_num_features)
    if not args.use_fake_dataset:
        dataset = FixedDataset(args.lr_dir, args.hr_dir)
    else:
        dataset = FakeDataset()
    trainer = Trainer(model, dataset, args.ckpt_dir, args.batch_size, args.learning_rate, device)

    for epoch in range(args.num_epochs):
        trainer.train_one_epoch()
        trainer.validate()
        trainer.save_model()
