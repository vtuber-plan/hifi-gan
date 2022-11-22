import torch
import torchaudio
import torchaudio.transforms as T

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import random

from ..mel_processing import hann_window

class AudioPipeline(torch.nn.Module):
    def __init__(
        self,
        freq=16000,
        n_fft=1024,
        n_mel=128,
        win_length=1024,
        hop_length=256
    ):
        super().__init__()

        self.freq=freq

        pad = int((n_fft-hop_length)/2)
        self.spec = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            pad=pad, power=None,center=False, pad_mode='reflect', normalized=False, onesided=True)

        # self.strech = T.TimeStretch(hop_length=hop_length, n_freq=freq)
        self.spec_aug = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=80),
            # T.TimeMasking(time_mask_param=80),
        )

        self.mel_scale = T.MelScale(n_mels=n_mel, sample_rate=freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor, aug: bool=False) -> torch.Tensor:
        shift_waveform = waveform
        # Convert to power spectrogram
        spec = self.spec(shift_waveform)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
        # Apply SpecAugment
        if aug:
            spec = self.spec_aug(spec)
        # Convert to mel-scale
        mel = self.mel_scale(spec)
        return mel