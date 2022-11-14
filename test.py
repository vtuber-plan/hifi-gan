import torch, torchaudio

from hifigan.mel_processing import mel_spectrogram_torch, spectrogram_torch_audio

# Load checkpoint
hifigan = torch.hub.load("vtuber-plan/hifi-gan:main", "hifigan_48k", force_reload=False).cuda()

# Load audio
wav, sr = torchaudio.load("test.wav")
assert sr == 48000

mel = mel_spectrogram_torch(wav, 2048, 256, 48000, 512, 2048, 0, None, False)

mel = mel.cuda()
out = hifigan(mel)

wav_out = out.squeeze(0).cpu()

torchaudio.save("test_out.wav", wav_out, sr)