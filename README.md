# Hifi-GAN
An 48kHz implementation of HiFi-GAN for Voice Conversion.


# Example

```Python
import torch, torchaudio
from hifigan.mel_processing import mel_spectrogram_torch
hifigan = torch.hub.load("vtuber-plan/hifi-gan:main", "hifigan_48k")
wav, sr = torchaudio.load("test.wav")
assert sr == 48000

mel = mel_spectrogram_torch(wav, 2048, 256, 48000, 512, 2048, 0, None, False)
mel = mel.cuda()
out = hifigan(mel)

wav_out = out.squeeze(0).cpu()
torchaudio.save("test_out.wav", wav_out, sr)
```

# Pretrained Model Info
|  Name            | Dataset   | Fine-tuned |
|  ----            | ----      |   ----     |
|  Hifi-GAN-48k    | Universal |     No     |

# Train Vocoder
Place all audio files under the dataset folder.
Then run these commands:
```bash
python filelist.py
python split.py
python train.py
```

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
