
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

import torchaudio
import torchaudio.transforms as T

from torch.nn import AvgPool1d

from .discriminator import DiscriminatorP, DiscriminatorS, DiscriminatorSpec

class SpectrogramDiscriminator(torch.nn.Module):
    def __init__(self, n_fft=1024, win_length=1024, hop_length=256, use_spectral_norm=False):
        super(SpectrogramDiscriminator, self).__init__()
        pad = int((n_fft-hop_length)/2)
        self.spec = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            pad=pad, power=None,center=False, pad_mode='reflect', normalized=False, onesided=True)

        self.discriminators = nn.ModuleList([
            DiscriminatorSpec(n_fft=n_fft, use_spectral_norm=use_spectral_norm),
            DiscriminatorSpec(n_fft=n_fft),
            DiscriminatorSpec(n_fft=n_fft),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(kernel_size=4, stride=2, padding=2),
            AvgPool1d(kernel_size=4, stride=2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_spec = self.spec(y)
        y_spec = torch.sqrt(y_spec.real.pow(2) + y_spec.imag.pow(2) + 1e-6).squeeze(1)

        y_hat_spec = self.spec(y_hat)
        y_hat_spec = torch.sqrt(y_hat_spec.real.pow(2) + y_hat_spec.imag.pow(2) + 1e-6).squeeze(1)

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y_spec = self.meanpools[i-1](y_spec)
                y_hat_spec = self.meanpools[i-1](y_hat_spec)
            y_d_r, fmap_r = d(y_spec)
            y_d_g, fmap_g = d(y_hat_spec)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
