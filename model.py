"""
Single-scale super-resolution neural model and its internal functional modules.
The model is a simplification of NAS's SingleNetwork.

References: https://github.com/kaist-ina/NAS_public
"""

__author__ = "Yihang Wu"

import math

import torch.nn as nn


class SingleNetwork(nn.Module):
    VALID_SCALES = (1, 2, 3, 4)  # the up-scaling factors of images

    def __init__(self, scale, num_blocks, num_channels, num_features, bias=True, activation=nn.ReLU(True)):
        """
        A single-scale single-image super-resolution neural model.

        Args:
            scale (int): up-scaling factor. The width of hr image is [scale] times larger than that of a lr image
            num_blocks (int): the number of residual blocks
            num_channels (int): the number of channels in an image
            num_features (int): the number of channels used throughout convolutional computations
            bias (bool): add bias or not
            activation (nn.Module): activate function
        """
        super(SingleNetwork, self).__init__()

        self.scale = scale
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.num_features = num_features

        if self.scale not in SingleNetwork.VALID_SCALES:
            raise NotImplementedError

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
        x = self.head(x)  # (*, num_features, input_height, input_width)

        res = x
        for i in range(self.num_blocks):
            res = self.body[i](res)
        res = self.body_end(res)
        res += x  # residual connection

        x = res  # (*, num_features, input_height, input_width)
        if self.scale > 1:
            x = self.upsampler(x)  # (*, num_features, target_height, target_width)

        x = self.tail(x)  # (*, num_channels, target_height, target_width)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, num_feats: int, bias: bool = True, batch_norm: bool = False, act: nn.Module = nn.ReLU(True),
                 residual_scale=1):
        """
        The residual block in SingalNetwork, which is a stack of Conv, ReLU, Conv and Sum.

        Args:
            num_feats (int): the number of channels of the convolutional kernel
            bias (bool): whether to use bias value
            batch_norm (bool): whether to use batch normalization
            act (nn.Module): activation function
            residual_scale (float): the factor to scale the residual
        """
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
        if self.residual_scale != 1:  # scale the residual
            res = self.block(x).mul(self.residual_scale)
        else:
            res = self.block(x)
        res += x  # residual connection

        return res


class Upsampler(nn.Module):
    def __init__(self, scale: int, num_feats: int, bias: bool = True, batch_norm: bool = False, act: nn.Module = None):
        """
        This module up-samples the inputs to target outputs in terms of a specified scaling factor

        Args:
            scale (int): scaling factor
            num_feats (int): the number of channels of the image
            bias (bool): whether to use bias value
            batch_norm (bool): whether to use batch normalization
            act (nn.Module): activation function
        """
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
