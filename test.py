import torch, torchaudio

from hifigan.mel_processing import mel_spectrogram_torch

# Load checkpoint (either hubert_soft or hubert_discrete)
hifigan = torch.hub.load("vtuber-plan/hifi-gan:main", "HifiGANGenerator").cuda()

# Load audio
wav, sr = torchaudio.load("test.wav")
assert sr == 48000
wav = wav.unsqueeze(0).cuda()

mel = mel_spectrogram_torch(wav, 2048, 256, 48000, 512, 2048, 0, None, False)

units = hifigan(mel)