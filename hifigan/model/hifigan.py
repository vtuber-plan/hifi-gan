
import itertools
import logging
from typing import Any, Dict
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchaudio
import torchaudio.transforms as T

import random

import pytorch_lightning as pl

from .discriminators.multi_scale_discriminator import MultiScaleDiscriminator
from .discriminators.multi_period_discriminator import MultiPeriodDiscriminator
from .generators.generator import Generator

from ..mel_processing import spec_to_mel_torch, mel_spectrogram_torch, spectrogram_torch, spectrogram_torch_audio
from .losses import discriminator_loss, kl_loss,feature_loss, generator_loss
from .. import commons
from .. import utils

class HifiGAN(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(*[k for k in kwargs])

        self.net_g = Generator()
        self.net_period_d = MultiPeriodDiscriminator(self.hparams.model.use_spectral_norm)
        self.net_scale_d = MultiScaleDiscriminator(self.hparams.model.use_spectral_norm)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, optimizer_idx: int):
        x_wav, x_wav_lengths = batch["x_wav_values"], batch["x_wav_lengths"]
        y_wav, y_wav_lengths = batch["y_wav_values"], batch["y_wav_lengths"]

        # generator forward
        y_hat, ids_slice, z_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = \
            self.net_g(x_wav, x_wav_lengths, x_pitch, x_pitch_lengths, y_spec, y_spec_lengths, sid=speakers)
        y = commons.slice_segments(y_wav, ids_slice * self.hparams.data.hop_length, self.hparams.train.segment_size) # slice 

        # Generator
        if optimizer_idx == 0:
            y_dp_hat_r, y_dp_hat_g, fmap_p_r, fmap_p_g = self.net_period_d(y, y_hat)
            loss_p_fm = feature_loss(fmap_p_r, fmap_p_g)
            loss_p_gen, losses_p_gen = generator_loss(y_dp_hat_g)

            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.net_scale_d(y, y_hat)
            loss_s_fm = feature_loss(fmap_s_r, fmap_s_g)
            loss_s_gen, losses_s_gen = generator_loss(y_ds_hat_g)

            y_spec_slice = commons.slice_segments(y_spec, ids_slice, self.hparams.train.segment_size // self.hparams.data.hop_length)

            y_spec_hat = spectrogram_torch_audio(y_hat,
                self.hparams.data.filter_length,
                self.hparams.data.target_sampling_rate,
                self.hparams.data.hop_length,
                self.hparams.data.win_length, center=False).squeeze(1)
            
            y_mel_hat = spec_to_mel_torch(
                y_spec_hat, 
                self.hparams.data.filter_length, 
                self.hparams.data.n_mel_channels, 
                self.hparams.data.target_sampling_rate,
                self.hparams.data.mel_fmin, 
                self.hparams.data.mel_fmax)
            
            mel = spec_to_mel_torch(
                y_spec, 
                self.hparams.data.filter_length, 
                self.hparams.data.n_mel_channels, 
                self.hparams.data.target_sampling_rate,
                self.hparams.data.mel_fmin, 
                self.hparams.data.mel_fmax)
            y_mel_slice = commons.slice_segments(mel, ids_slice, self.hparams.train.segment_size // self.hparams.data.hop_length)

            # kl
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hparams.train.c_kl
            # spec
            # loss_spec = F.l1_loss(y_spec_hat, y_spec_slice) * self.hparams.train.c_mel
            # mel
            loss_mel = F.l1_loss(y_mel_hat, y_mel_slice) * self.hparams.train.c_mel

            loss_gen_all = (loss_s_gen + loss_s_fm) + (loss_p_gen + loss_p_fm) + loss_mel + loss_kl

            grad_norm_g = commons.clip_grad_value_(self.net_g.parameters(), None)

            # Logging to TensorBoard by default
            lr = self.optim_g.param_groups[0]['lr']
            scalar_dict = {"loss/g/total": loss_gen_all, "learning_rate": lr, "grad_norm_g": grad_norm_g}
            scalar_dict.update({
                "loss/g/p_fm": loss_p_fm,
                "loss/g/s_fm": loss_s_fm,
                "loss/g/p_gen": loss_p_gen,
                "loss/g/s_gen": loss_s_gen,
                "loss/g/loss_mel": loss_mel,
            })

            # scalar_dict.update({"loss/g/p_gen_{}".format(i): v for i, v in enumerate(losses_p_gen)})
            # scalar_dict.update({"loss/g/s_gen_{}".format(i): v for i, v in enumerate(losses_s_gen)})

            # image_dict = { 
            #     "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            #     "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            #     "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())
            # }
            image_dict = {}
            
            tensorboard = self.logger.experiment
            utils.summarize(
                writer=tensorboard,
                global_step=self.global_step, 
                images=image_dict,
                scalars=scalar_dict)
            return loss_gen_all

        # Discriminator
        if optimizer_idx == 1:
            # MPD
            y_dp_hat_r, y_dp_hat_g, _, _ = self.net_period_d(y, y_hat.detach())
            loss_disc_p, losses_disc_p_r, losses_disc_p_g = discriminator_loss(y_dp_hat_r, y_dp_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = self.net_scale_d(y, y_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_p + loss_disc_s

            grad_norm_p_d = commons.clip_grad_value_(self.net_period_d.parameters(), None)
            grad_norm_s_d = commons.clip_grad_value_(self.net_scale_d.parameters(), None)

            # log
            lr = self.optim_g.param_groups[0]['lr']
            scalar_dict = {"loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_p_d": grad_norm_p_d, "grad_norm_s_d": grad_norm_s_d}
            scalar_dict.update({"loss/d_p_r/{}".format(i): v for i, v in enumerate(losses_disc_p_r)})
            scalar_dict.update({"loss/d_p_g/{}".format(i): v for i, v in enumerate(losses_disc_p_g)})
            scalar_dict.update({"loss/d_s_r/{}".format(i): v for i, v in enumerate(losses_disc_s_r)})
            scalar_dict.update({"loss/d_s_g/{}".format(i): v for i, v in enumerate(losses_disc_s_g)})

            image_dict = {}
            
            tensorboard = self.logger.experiment

            utils.summarize(
                writer=tensorboard,
                global_step=self.global_step, 
                images=image_dict,
                scalars=scalar_dict)

            return loss_disc_all

    def validation_step(self, batch, batch_idx):
        self.net_g.eval()
        
        speakers = batch.get("sid", None)

        x_wav, x_wav_lengths = batch["x_wav_values"], batch["x_wav_lengths"]
        x_pitch, x_pitch_lengths = batch["x_pitch_values"], batch["x_pitch_lengths"]

        y_wav, y_wav_lengths = batch["y_wav_values"], batch["y_wav_lengths"]

        y_spec = spectrogram_torch_audio(y_wav.squeeze(1),
            self.hparams.data.filter_length,
            self.hparams.data.target_sampling_rate,
            self.hparams.data.hop_length,
            self.hparams.data.win_length, center=False)
        y_spec_lengths = (y_wav_lengths / self.hparams.data.hop_length).long()

        # remove else
        y_spec = y_spec[:1]
        y_spec_lengths = y_spec_lengths[:1]
        
        len_scale = (self.hparams.data.target_sampling_rate / self.hparams.data.hop_length) / self.hparams.data.source_sampling_rate
        y_hat, mask, (z, z_p, m_p, logs_p) = self.net_g.infer(
            x_wav, x_wav_lengths, x_pitch, x_pitch_lengths,
            sid=speakers, length_scale=len_scale, max_len=1000)
        y_hat_lengths = mask.sum([1,2]).long() * self.hparams.data.hop_length

        mel = spec_to_mel_torch(
            y_spec, 
            self.hparams.data.filter_length, 
            self.hparams.data.n_mel_channels, 
            self.hparams.data.target_sampling_rate,
            self.hparams.data.mel_fmin, 
            self.hparams.data.mel_fmax)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            self.hparams.data.filter_length,
            self.hparams.data.n_mel_channels,
            self.hparams.data.target_sampling_rate,
            self.hparams.data.hop_length,
            self.hparams.data.win_length,
            self.hparams.data.mel_fmin,
            self.hparams.data.mel_fmax
        )
        image_dict = {
            "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
        }
        audio_dict = {
            "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
        }
        # if self.global_step == 0:
        image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
        audio_dict.update({"gt/audio": y_wav[0,:,:y_wav_lengths[0]]})

        tensorboard = self.logger.experiment
        utils.summarize(
            writer=tensorboard,
            global_step=self.global_step, 
            images=image_dict,
            audios=audio_dict,
            audio_sampling_rate=self.hparams.data.target_sampling_rate)

    def configure_optimizers(self):
        self.optim_g = torch.optim.AdamW(
            self.net_g.parameters(), 
            self.hparams.train.learning_rate, 
            betas=self.hparams.train.betas, 
            eps=self.hparams.train.eps)
        self.optim_d = torch.optim.AdamW(
            itertools.chain(self.net_period_d.parameters(), self.net_scale_d.parameters()),
            self.hparams.train.learning_rate, 
            betas=self.hparams.train.betas, 
            eps=self.hparams.train.eps)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=self.hparams.train.lr_decay)
        self.scheduler_g.last_epoch = self.current_epoch - 1
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=self.hparams.train.lr_decay)
        self.scheduler_d.last_epoch = self.current_epoch - 1

        return [self.optim_g, self.optim_d], [self.scheduler_g, self.scheduler_d]
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logging.info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                logging.info(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)