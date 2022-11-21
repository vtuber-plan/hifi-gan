import torch
import os
import glob


import torchaudio
import torchaudio.transforms as T

class AudioPipeline(torch.nn.Module):
    def __init__(
        self,
        freq=16000,
        n_fft=1024,
        n_mel=128,
        win_length=1024,
        hop_length=256,
    ):
        super().__init__()
        self.freq=freq
        pad = int((n_fft-hop_length)/2)
        self.spec = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            pad=pad, power=None,center=False, pad_mode='reflect', normalized=False, onesided=True)

        self.mel_scale = T.MelScale(n_mels=n_mel, sample_rate=freq, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        shift_waveform = waveform
        # Convert to power spectrogram
        spec = self.spec(shift_waveform)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
        # Convert to mel-scale
        mel = self.mel_scale(spec)
        return mel

def load_local():
    from hifigan.model.hifigan import HifiGAN

    ckpt_path = None
    if os.path.exists("logs/lightning_logs"):
        versions = glob.glob("logs/lightning_logs/version_*")
        if len(list(versions)) > 0:
            last_ver = sorted(list(versions), key=lambda p: int(p.split("_")[-1]))[-1]
            last_ckpt = os.path.join(last_ver, "checkpoints/last.ckpt")
            if os.path.exists(last_ckpt):
                ckpt_path = last_ckpt
    
    print(ckpt_path)
    
    model = HifiGAN.load_from_checkpoint(checkpoint_path=ckpt_path, strict=False)

    return model.net_g

def load_remote():
    return torch.hub.load("vtuber-plan/hifi-gan:v0.2.1", "hifigan_48k", force_reload=True)

device = "cpu"

# Load Remote checkpoint
# hifigan = load_remote().to(device)

# Load Local checkpoint
hifigan = load_local().to(device)

# Load audio
wav, sr = torchaudio.load("zszy_48k.wav")
assert sr == 48000

# mel = mel_spectrogram_torch(wav, 2048, 128, 48000, 512, 2048, 0, None, False)
audio_pipeline = AudioPipeline(freq=48000,
                                n_fft=2048,
                                n_mel=128,
                                win_length=2048,
                                hop_length=512)
mel = audio_pipeline(wav)
out = hifigan(mel)

wav_out = out.squeeze(0).cpu()

torchaudio.save("test_out.wav", wav_out, sr)

# import librosa
# import scipy
# audio = librosa.feature.inverse.mel_to_audio(mel.detach().cpu().numpy(), sr=sr, n_fft=2048, hop_length=512, win_length=2048)
# torchaudio.save("test_out2.wav", torch.tensor(audio), sr)