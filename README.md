# Hifi-GAN
An 48kHz implementation of HiFi-GAN for Voice Conversion.


# Example

```Python
import torch
import os
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

device = "cpu"

hifigan = torch.hub.load("vtuber-plan/hifi-gan:v0.3.0", "hifigan_48k", force_reload=True).to(device)

# Load audio
wav, sr = torchaudio.load("test.wav")
assert sr == 48000

audio_pipeline = AudioPipeline(freq=48000,
                                n_fft=2048,
                                n_mel=128,
                                win_length=2048,
                                hop_length=512)
mel = audio_pipeline(wav)
out = hifigan(mel)

wav_out = out.squeeze(0).cpu()

torchaudio.save("test_out.wav", wav_out, sr)
```

# Pretrained Model Info
|  Name            | Dataset   | Fine-tuned |
|  ----            | ----      |   ----     |
|  Hifi-GAN-48k    | Universal |     No     |
|  Hifi-GAN-44.1k  | Universal |     No     |
|  Hifi-GAN-36k    | Universal |     No     |
|  Hifi-GAN-24k    | Universal |     No     |
|  Hifi-GAN-16k    | Universal |     No     |

Training Datasets: VCTK, JSUT and RAVDESS

# Train Vocoder
Place all audio files under the dataset folder.
Then run these commands:
```bash
python filelist.py
python split.py
python train.py
```

## Single GPU Training

## Single Node Multi GPUs Training

## Multi Node Multi GPUs Training

## TPU Training

# LICENSE
```
MIT License

Copyright (c) 2022 Vtuber Plan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

# Credits
Hifi-GAN models from [jik876](https://github.com/jik876/hifi-gan).

Inspired by [SOVITS](https://github.com/innnky/so-vits-svc) and [Diff-SVC](https://github.com/prophesier/diff-svc).
