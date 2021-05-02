"""
An offline training example.
"""

__author__ = "Yihang Wu"

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from model import SingleNetwork
from dataset import FixedDataset


class Trainer:
    def __init__(self, model, dataset, ckpt_dir):
        self.model = model
        self.dataset = dataset
        self.ckpt_dir = ckpt_dir

        self.dataloader = DataLoader(dataset=self.dataset, batch_size=32, num_workers=4,
                                     shuffle=True)

        self.device = torch.device('cuda')
        self.epoch = 0

        self.optimizer = optim.Adam(model.parameters(), lr=1e-4, )
        self.loss_func = nn.L1Loss()

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
    LR_DIR = 'data/360p'
    HR_DIR = 'data/720p'
    CKPT_DIR = 'data/pretrained'

    os.makedirs(CKPT_DIR, exist_ok=True)

    model = SingleNetwork(2, 6, 3, 6)
    dataset = FixedDataset(LR_DIR, HR_DIR)
    trainer = Trainer(model, dataset, CKPT_DIR)

    for epoch in range(20):
        trainer.train_one_epoch()
        trainer.validate()
    trainer.save_model()
