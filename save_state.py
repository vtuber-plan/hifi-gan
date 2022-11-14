import os
import glob
import torch
from hifigan.model.hifigan import HifiGAN
def main():
    ckpt_path = None
    if os.path.exists("logs/lightning_logs"):
        versions = glob.glob("logs/lightning_logs/version_*")
        if len(list(versions)) > 0:
            last_ver = sorted(list(versions))[-1]
            last_ckpt = os.path.join(last_ver, "checkpoints/last.ckpt")
            if os.path.exists(last_ckpt):
                ckpt_path = last_ckpt
    
    print(ckpt_path)
    
    model = HifiGAN.load_from_checkpoint(checkpoint_path=ckpt_path, strict=False)

    torch.save(model.net_g.state_dict(), "out.pt")

if __name__ == "__main__":
    main()