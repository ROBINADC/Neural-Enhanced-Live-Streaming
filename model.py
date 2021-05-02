"""
Neural networks and functional modules
"""

__author__ = "Yihang Wu"

import torch.nn as nn
import math

VALID_SCALES = (1, 2, 3, 4)


class SingleNetwork(nn.Module):
    def __init__(self, scale, num_blocks, num_channels, num_features, bias=True, activation=nn.ReLU(True)):
        super(SingleNetwork, self).__init__()

        self.scale = scale
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.num_features = num_features

        assert self.scale in VALID_SCALES

        # No early-exit implemented

        # Head of a single NAS model
        self.head = nn.Sequential(nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_features,
                                            kernel_size=3, stride=1, padding=1, bias=bias))

        # Body of model - consecutive residual blocks
        self.body = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.body.append(nn.Sequential(ResidualBlock(num_feats=self.num_features, bias=bias, act=activation)))

        self.body_end = nn.Sequential(nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features,
                                                kernel_size=3, stride=1, padding=1, bias=bias))

        # Upsampling
        if self.scale > 1:
            self.upsampler = nn.Sequential(Upsampler(scale=self.scale, num_feats=self.num_features, bias=bias))

        # Tail of model
        self.tail = nn.Sequential(nn.Conv2d(in_channels=self.num_features, out_channels=self.num_channels,
                                            kernel_size=3, stride=1, padding=1, bias=bias))

    def forward(self, x):
        """

        input shape (*, num_channels, input_height, input_width)
        output shape (*, num_channels, target_height, target_width)
        """
        x = self.head(x)

        res = x
        for i in range(self.num_blocks):
            res = self.body[i](res)
        res = self.body_end(res)
        res += x  # residual connection

        x = res
        if self.scale > 1:
            x = self.upsampler(x)

        x = self.tail(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, num_feats: int, bias: bool = True, batch_norm: bool = False, act: nn.Module = nn.ReLU(True),
                 residual_scale=1):
        super(ResidualBlock, self).__init__()
        modules = []

        for i in range(2):
            modules.append(nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=3, stride=1, padding=1, bias=bias))
            if batch_norm:
                modules.append(nn.BatchNorm2d(num_feats))
            if i == 0:
                modules.append(act)

        self.block = nn.Sequential(*modules)
        self.residual_scale = residual_scale

    def forward(self, x):
        if self.residual_scale != 1:
            res = self.block(x).mul(self.residual_scale)
        else:
            res = self.block(x)
        res += x  # residual connection

        return res


class Upsampler(nn.Module):
    def __init__(self, scale: int, num_feats: int, bias: bool = True, batch_norm: bool = False, act: nn.Module = None):
        super(Upsampler, self).__init__()

        modules = []
        if scale & (scale - 1) == 0:  # scale = 1, 2, 4
            for _ in range(int(math.log(scale, 2))):
                modules.append(nn.Conv2d(in_channels=num_feats, out_channels=4 * num_feats, kernel_size=3, stride=1, padding=1, bias=bias))
                modules.append(nn.PixelShuffle(2))

                if batch_norm:
                    modules.append(nn.BatchNorm2d(num_feats))
                if act:
                    modules.append(act)
        elif scale == 3:
            modules.append(nn.Conv2d(in_channels=num_feats, out_channels=9 * num_feats, kernel_size=3, stride=1, padding=1, bias=bias))
            modules.append(nn.PixelShuffle(3))

            if batch_norm:
                modules.append(nn.BatchNorm2d(num_feats))
            if act:
                modules.append(act)
        else:
            raise NotImplementedError

        self.upsampler = nn.Sequential(*modules)

    def forward(self, x):
        return self.upsampler(x)


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader
    from dataset import DummyDataset

    model = SingleNetwork(2, 6, 3, 6)

    model.to('cuda')
    for i, (xs, ys) in enumerate(DataLoader(DummyDataset(), 4)):
        xs, ys = xs.to('cuda'), ys.to('cuda')

        outs = model(xs)
        print(xs.size(), ys.size(), outs.size())
        break
