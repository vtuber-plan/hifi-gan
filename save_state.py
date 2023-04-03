import os
import glob
import shutil
from typing import Optional
import torch
from hifigan.light.hifigan import HifiGAN
import hashlib

def save(ckpt_path: str, name: str):
    model = HifiGAN.load_from_checkpoint(checkpoint_path=ckpt_path, strict=True)
    # print(model.net_g.state_dict())
    torch.save(model.net_g.state_dict(), f"hifigan-{name}-net-g.pt")
    torch.save(model.net_period_d.state_dict(), f"hifigan-{name}-net-period-d.pt")
    torch.save(model.net_scale_d.state_dict(), f"hifigan-{name}-net-scale-d.pt")

    h = hashlib.md5(open(f"hifigan-{name}-net-g.pt",'rb').read()).hexdigest()
    shutil.move(f"hifigan-{name}-net-g.pt", f"hifigan-{name}-net-g-{h}.pt")
    h = hashlib.md5(open(f"hifigan-{name}-net-period-d.pt",'rb').read()).hexdigest()
    shutil.move(f"hifigan-{name}-net-period-d.pt", f"hifigan-{name}-net-period-d-{h}.pt")
    h = hashlib.md5(open(f"hifigan-{name}-net-scale-d.pt",'rb').read()).hexdigest()
    shutil.move(f"hifigan-{name}-net-scale-d.pt", f"hifigan-{name}-net-scale-d-{h}.pt")
    
def last_checkpoint(path: str) -> Optional[str]:
    ckpt_path = None
    if os.path.exists(os.path.join(path, "lightning_logs")):
        versions = glob.glob(os.path.join(path, "lightning_logs", "version_*"))
        if len(list(versions)) > 0:
            last_ver = sorted(list(versions), key=lambda p: int(p.split("_")[-1]))[-1]
            last_ckpt = os.path.join(last_ver, "checkpoints/last.ckpt")
            if os.path.exists(last_ckpt):
                ckpt_path = last_ckpt
    return ckpt_path

import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default="./logs", help='Loggin Path')
    parser.add_argument('-n', '--name', type=str, default="48k", help='sr')
    args = parser.parse_args()
    ckpt_path = last_checkpoint(args.dir)
    print(ckpt_path)
    save(ckpt_path, args.name)
if __name__ == "__main__":
    main()