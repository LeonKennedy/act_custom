#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: predict_diffusion.py
@time: 2024/7/26 17:44
@desc:
"""
import argparse
import collections
import os

import numpy as np
import torch
from tqdm.auto import tqdm
from data import EpisodicDataset, normalize_data, unnormalize_data
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from constant import dataset_path, pred_horizon, obs_horizon, action_horizon, action_dim
from policy_diffusion import build_policy

num_diffusion_iters = 100

# dataset = PushTImageDataset(
#     dataset_path=dataset_path,
#     pred_horizon=pred_horizon,
#     obs_horizon=obs_horizon,
#     action_horizon=action_horizon
# )
#
# stats = dataset.stats

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


def get_stats(data_path: str):
    data_set = EpisodicDataset(data_path)
    return data_set.stats


def get_obs():
    return np.zeros((1, 2, 14)), np.ones((1, 2, 4, 3, 240, 320))


def done(action):
    pass


def build_and_load_policy(obs_horizon, action_dim, ckpt_path: str):
    if os.path.exists(ckpt_path):
        params = torch.load(ckpt_path)
        policy = build_policy(obs_horizon, action_dim, params['stats'], params['weights'])
        return policy
    raise FileNotFoundError("ckpt not found")


def predict(args):
    policy = build_and_load_policy(args.obs_horizon, args.action_dim, args.ckpt)

    obs, obs_images = get_obs()
    step_idx = 0
    while 1:
        action = policy.inference(obs_images, obs)
        # (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):
            # stepping env
            done(action)
            # and reward/vis

            # update progress bar
            step_idx += 1

    # print out the maximum target coverage
    print('Score: ', max(rewards))

    # visualize
    # from IPython.display import Video
    vwrite('vis.mp4', imgs)


if __name__ == "__main__":
    device = torch.device('cuda')
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', action='store', type=str, help='ckpt dir', default="diff_ckpt/policy_epoch_best.ckpt")
    parser.add_argument('--data_path', action='store', type=str, help='data path', default='./data/train.zarr')

    # for DIFFUSION
    parser.add_argument('--action_dim', action='store', type=int, help='action dim', default=14)
    parser.add_argument('--obs_horizon', action='store', type=int, help='obs horizon', default=2)

    predict(parser.parse_args())
