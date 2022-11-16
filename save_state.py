import os
import glob
import torch
from hifigan.model.hifigan import HifiGAN

def save(ckpt_path: str):
    model = HifiGAN.load_from_checkpoint(checkpoint_path=ckpt_path, strict=True)
    # print(model.net_g.state_dict())
    torch.save(model.net_g.state_dict(), "net_g.pt")
    torch.save(model.net_period_d.state_dict(), "net_period_d.pt")
    torch.save(model.net_scale_d.state_dict(), "net_scale_d.pt")

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
    save("logs/lightning_logs_v1/version_8/checkpoints/last.ckpt")
if __name__ == "__main__":
    main()