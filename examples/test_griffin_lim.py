
import torchaudio
import torch
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

# Load audio
wav, sr = torchaudio.load("zszy_48k.wav")
assert sr == 48000

audio_pipeline = AudioPipeline(freq=48000,
                                n_fft=2048,
                                n_mel=128,
                                win_length=2048,
                                hop_length=512)

mel = audio_pipeline(wav)

import librosa
import scipy
audio = librosa.feature.inverse.mel_to_audio(mel.detach().cpu().numpy(), sr=sr, n_fft=2048, hop_length=512, win_length=2048, center=False)
torchaudio.save("test_out2.wav", torch.tensor(audio), sr)