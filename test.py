import torch, torchaudio
import os
import glob
from hifigan.mel_processing import mel_spectrogram_torch, spectrogram_torch_audio
from hifigan.model.hifigan import HifiGAN

def load_local():
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
    return torch.hub.load("vtuber-plan/hifi-gan:main", "hifigan_48k", force_reload=True)

device = "cpu"

# Load Remote checkpoint
# hifigan = load_remote().to(device)

# Load Local checkpoint
hifigan = load_local().to(device)

# Load audio
wav, sr = torchaudio.load("zszy_48k.wav")
assert sr == 48000

mel = mel_spectrogram_torch(wav, 2048, 256, 48000, 512, 2048, 0, None, False)

mel = mel.to(device)
out = hifigan(mel)

wav_out = out.squeeze(0).cpu()

torchaudio.save("test_out.wav", wav_out, sr)

# import librosa
# import scipy
# audio = librosa.feature.inverse.mel_to_audio(mel.detach().cpu().numpy(), sr=sr, n_fft=2048, hop_length=512, win_length=2048)
# torchaudio.save("test_out2.wav", torch.tensor(audio), sr)