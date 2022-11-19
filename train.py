

import os
import json
import glob
import argparse
from typing import Optional
import torch
import torchaudio
import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from hifigan.model.hifigan import HifiGAN

from hifigan.data.collate import MelCollate

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

from hifigan.hparams import HParams
from hifigan.data.dataset import MelDataset, MelDataset
from hifigan.utils import load_state_dict

def get_hparams(config_path: str) -> HParams:
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    
    hparams = HParams(**config)
    return hparams

def last_checkpoint() -> Optional[str]:
    ckpt_path = None
    if os.path.exists("logs/lightning_logs"):
        versions = glob.glob("logs/lightning_logs/version_*")
        if len(list(versions)) > 0:
            last_ver = sorted(list(versions), key=lambda p: int(p.split("_")[-1]))[-1]
            last_ckpt = os.path.join(last_ver, "checkpoints/last.ckpt")
            if os.path.exists(last_ckpt):
                ckpt_path = last_ckpt
    return ckpt_path
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/48k.json", help='JSON file for configuration')
    parser.add_argument('-a', '--accelerator', type=str, default="gpu", help='training device')
    parser.add_argument('-d', '--device', type=str, default="6", help='training device ids')
    args = parser.parse_args()

    hparams = get_hparams(args.config)
    pl.utilities.seed.seed_everything(hparams.train.seed)

    devices = [int(n.strip()) for n in args.device.split(",")]

    checkpoint_callback = ModelCheckpoint(dirpath=None, save_last=True, every_n_train_steps=2000, save_weights_only=False)
    earlystop_callback = EarlyStopping(monitor="valid/loss_mel_epoch", mode="min", patience=7)

    trainer_params = {
        "accelerator": args.accelerator,
        "callbacks": [checkpoint_callback, earlystop_callback],
    }

    if args.accelerator != "cpu":
        trainer_params["devices"] = devices

    if len(devices) > 1:
        trainer_params["strategy"] = "ddp"

    trainer_params.update(hparams.trainer)

    if hparams.train.fp16_run:
        trainer_params["amp_backend"] = "native"
        trainer_params["precision"] = 16

    # data
    train_dataset = MelDataset(hparams.data.training_files, hparams.data)
    valid_dataset = MelDataset(hparams.data.validation_files, hparams.data)
        
    collate_fn = MelCollate()

    if "strategy" in trainer_params and trainer_params["strategy"] == "ddp":
        batch_per_gpu = hparams.train.batch_size // len(devices)
    else:
        batch_per_gpu = hparams.train.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_per_gpu, num_workers=4, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=4, num_workers=4, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    # model
    model = HifiGAN(**hparams)
    # model = HifiGAN.load_from_checkpoint(checkpoint_path="logs/lightning_logs_v1/version_8/checkpoints/last.ckpt", strict=True)
    # model.net_g._load_from_state_dict(torch.load("net_g.pt"), strict=False)
    # model.net_period_d._load_from_state_dict(torch.load("net_period_d.pt"), strict=False)
    # model.net_scale_d._load_from_state_dict(torch.load("net_scale_d.pt"), strict=False)

    # profiler = AdvancedProfiler(filename="profile.txt")
    trainer = pl.Trainer(**trainer_params) # , profiler=profiler, max_steps=200
    # resume training
    ckpt_path = last_checkpoint()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
  main()
