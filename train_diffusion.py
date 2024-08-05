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
import os

import torch
import torch.nn as nn

from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import numpy as np

from data import build_dataloader
from policy_diffusion import build_policy


def run(args):
    print(args)
    obs_horizon = args.obs_horizon
    action_dim = args.action_dim
    pred_horizon = args.pred_horizon
    num_epochs = args.num_epochs
    iter_num = args.iter_num
    camera_cnt = 3

    dataloader, stats = build_dataloader(args.data_path, args.batch_size, obs_horizon, pred_horizon)
    if os.path.exists(args.ckpt):
        print("load weight from", args.ckpt)
        params = torch.load(args.ckpt)
        obs_horizon = params["obs_horizon"]
        iter_num = params["iter_num"]
        policy = build_policy(obs_horizon, action_dim, camera_cnt, iter_num, stats, params['weights'])
        min_loss = params["loss"]
        current_epoch = params["epoch"]
    else:
        current_epoch = 0
        min_loss = np.inf
        policy = build_policy(obs_horizon, action_dim, camera_cnt, iter_num, stats)
    policy.create_ema()
    optimizer = torch.optim.AdamW(params=policy.nets.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    with tqdm(range(current_epoch, current_epoch + num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = list()
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch['image'][:, :obs_horizon].to(device)  # (B
                    nagent_pos = nbatch['agent_pos'][:, :obs_horizon].to(device)  # (B)
                    naction = nbatch['action'].to(device)  # (B

                    noise_pred, noise = policy.forward(nimage, nagent_pos, naction)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                    policy.ema_step()
                    loss_cpu = loss.item()

                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            current_loss = np.mean(epoch_loss)
            if current_loss < min_loss:
                min_loss = current_loss
                policy.save(os.path.join(args.ckpt_dir, f'policy_epoch_best.ckpt'), current_loss, epoch_idx, obs_horizon,
                            pred_horizon)
            if epoch_idx > 0 and epoch_idx % 100 == 0:
                policy.save(os.path.join(args.ckpt_dir, f'policy_epoch_{epoch_idx}.ckpt'), current_loss, epoch_idx,
                            obs_horizon, pred_horizon)
            tglobal.set_postfix(loss=current_loss)


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', action='store', type=str, help='ckpt dir', default='')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt dir', default="./diff_ckpt")
    parser.add_argument('--batch_size', action='store', type=int, help='batch size', default=64)
    parser.add_argument('--num_epochs', action='store', type=int, help='num epochs', default=100)
    parser.add_argument('--data_path', action='store', type=str, help='data path', default='./data/train.zarr')

    # for DIFFUSION
    parser.add_argument('--action_dim', action='store', type=int, help='action dim', default=14)
    parser.add_argument('--pred_horizon', action='store', type=int, help='action horizon', default=16)
    parser.add_argument('--obs_horizon', action='store', type=int, help='obs horizon', default=3)
    parser.add_argument("--iter_num", action='store', type=int, help='iter num', default=70)

    run(parser.parse_args())
