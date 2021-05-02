"""
Dataset classes for neural network.
"""

__author__ = "Yihang Wu"

import glob
import logging

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor

from misc import alphanum

logger = logging.getLogger(__name__)


class ExtendableDataset(Dataset):
    transform = Compose([ToTensor(), ])

    def __init__(self, num_items_per_epoch: int):
        """
        An extendable dataset.
        The dataset can be extended during the interval of two training epochs.
        The dataset has an assigned fixed-size training epoch,
        from which the items are sampled from all the existing data.
        The sampler sample data under uniform distribution.

        Examples:
            ExtendableDataset(3000)

        Args:
            num_items_per_epoch (int): the number of items within each training epoch.
        """
        super().__init__()

        self.num_items_per_epoch = num_items_per_epoch

        self._size = 0
        self._xs = []
        self._ys = []

        self._prob = None

    def extend(self, data: list) -> None:
        """
        Extend the internal dataset with a list of data.

        Args:
            data (list): a list of (x, y)
        """
        for x, y in data:
            self._xs.append(ExtendableDataset.transform(x))
            self._ys.append(ExtendableDataset.transform(y))
        self._size += len(data)

        self.update_prob()

        if self._prob is not None and len(self._prob) != self._size:
            raise ValueError(f'The length of probability distribution ({len(self._prob)}) '
                             f'is different from the dataset size ({self._size})')

    def update_prob(self) -> None:
        """
        A hook function to update the probability distribution for existing data if applicable.
        """
        pass

    @property
    def size(self):
        return self._size

    @property
    def prob(self):
        return self._prob

    @prob.setter
    def prob(self, probability):
        """
        Set the probability distribution for existing data.
        """
        self._prob = probability

    def __len__(self):
        if self._size == 0:
            raise ValueError
        return self.num_items_per_epoch

    def __getitem__(self, item):
        index = np.random.choice(self._size, p=self._prob)
        return self._xs[index], self._ys[index]


class RecentBiasDataset(ExtendableDataset):
    def __init__(self, num_items_per_epoch=3000, num_biased_samples=150, bias_weight=4):
        """
        RecentBiasDataset exposes the recent data in a higher weight.

        Examples:
            RecentBiasDataset(num_items_per_epoch=3000, num_biased_samples=150, bias_weight=4)
            will create an extendable dataset.
            The items are sampled in terms of weight.
            The dataset will give 4 times the weight to recent 150 items than remaining old items.

        Args:
            num_items_per_epoch (int): the number of items within each training epoch.
            num_biased_samples (int): the number of recent samples that have higher sample weight
            bias_weight (int): the probability of getting recent items is bias_weight times more than the old items
        """
        super().__init__(num_items_per_epoch)

        self.num_biased_samples = num_biased_samples
        self.bias_weight = bias_weight

    def update_prob(self):
        if self.size <= self.num_biased_samples:
            self.prob = None
        else:
            base = self.size + (self.bias_weight - 1) * self.num_biased_samples  # denominator
            old = 1 / base  # probability for old elements
            new = self.bias_weight / base  # probability for new elements
            self.prob = [old for _ in range(self.size - self.num_biased_samples)] + [new for _ in range(self.num_biased_samples)]


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


class DummyDataset(Dataset):
    """
    A dummy dataset that trivial data
    """

    def __init__(self, total=100):
        super(DummyDataset, self).__init__()

        self.total = total

    def __getitem__(self, index):
        x = torch.rand(3, 360, 640)
        y = torch.rand(3, 720, 1280)
        return x, y

    def __len__(self):
        return self.total


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader

    ExtendableDataset.transform = lambda x: x
    # dataset = ExtendableDataset(50)
    # dataloader = DataLoader(dataset, batch_size=20)
    # for i in range(10):  # 10 iters
    #     dataset.extend([(i, i) for _ in range(20)])
    #     print('new epoch')
    #     for x, y in dataloader:
    #         print(x)

    dataset = RecentBiasDataset(50, 20, 4)
    dataloader = DataLoader(dataset, batch_size=20)
    for i in range(10):  # 10 iters
        dataset.extend([(i, i) for _ in range(20)])
        print('new epoch')
        for x, y in dataloader:
            print(x)

    # dataset = FixedDataset('data/360p', 'data/1080p')
    # dataloader = DataLoader(dataset, batch_size=64)
    #
    # for x, y in dataloader:
    #     print(x.shape, y.shape)
    #     break
