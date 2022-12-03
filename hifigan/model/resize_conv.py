

from functools import reduce
import operator
from typing import List, Union
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from .commons import init_weights, get_padding

class ResizeConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResizeConv1d, self).__init__()
        self.stride = stride
        self.conv = Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding='same')
        self.conv.apply(init_weights)

        self.weight = self.conv.weight
        
    def forward(self, x):
        interpolated_x = torch.nn.functional.interpolate(x, size=(self.stride * x.shape[2],), mode='linear', align_corners=True)
        return self.conv(interpolated_x)

class ResizeConv1dBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResizeConv1dBlock, self).__init__()
        self.conv = weight_norm(ResizeConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride))
        self.weight = self.conv.weight
    def forward(self, x):
        return self.conv(x)

    def remove_weight_norm(self):
        for l in self.conv:
            remove_weight_norm(l)
