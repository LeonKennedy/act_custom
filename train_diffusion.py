#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: train_diffusion.py
@time: 2024/7/26 10:30
@desc:
"""
import argparse

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import numpy as np

from data import build_dataloader
from policy_diffusion import build_net


def run(args):
    print(args)
    obs_horizon = args.obs_horizon
    num_diffusion_iters = 100
    num_epochs = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )
    dataloader = build_dataloader(args.data_path)
    nets = build_net(args.obs_horizon, args.action_dim)
    _ = nets.to(device)
    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch['image'][:, :obs_horizon, 0].to(device)
                    nagent_pos = nbatch['agent_pos'][:, :obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    # encoder vision features
                    image_features = nets['vision_encoder'](nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(
                        *nimage.shape[:2], -1)
                    # (B,obs_horizon,D)

                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = nets['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt dir', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch size', default=64)
    parser.add_argument('--num_epochs', action='store', type=int, help='num epochs', default=100)
    parser.add_argument('--data_path', action='store', type=str, help='data path', default='./data/train.zarr')

    # for DIFFUSION
    parser.add_argument('--action_dim', action='store', type=int, help='action dim', default=14)
    parser.add_argument('--obs_horizon', action='store', type=int, help='obs horizon', default=2)

    run(parser.parse_args())
