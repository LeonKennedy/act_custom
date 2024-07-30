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
import time

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm
from loguru import logger

from camera import CameraGroup
from data import EpisodicDataset, normalize_data, unnormalize_data
from dr import build_two_arm
# from constant import dataset_path, pred_horizon, obs_horizon, action_horizon, action_dim
from policy_diffusion import build_policy, get_noise_ddpm_schedule
from task_config import TASK_CONFIG

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


def build_and_load_policy(action_dim, ckpt_path: str):
    if os.path.exists(ckpt_path):
        params = torch.load(ckpt_path)
        obs_horizon = 2
        action_horizon = 8
        policy = build_policy(obs_horizon, action_dim, params['stats'], params['weights'])
        return policy, obs_horizon, action_horizon
    raise FileNotFoundError("ckpt not found")


class Robo:
    def __init__(self, obs_horizon: int):
        self.arm_left, self.arm_right = build_two_arm(TASK_CONFIG["Pick_Cube"])
        self.camera = CameraGroup()
        self.angles = collections.deque(maxlen=obs_horizon)
        self.images = collections.deque(maxlen=obs_horizon)
        self.step_idx = 0

    def start(self):
        # free master
        self.arm_left.master.free()
        self.arm_right.master.free()
        self.arm_left.puppet.move_to1([0, -10, -90, -20, 90, 0])
        self.arm_right.puppet.move_to1([0, 0, 90, 0, -86, 0])

    def get_obs(self):
        angle = np.stack(self.angles)
        image = np.stack(self.images)
        image = np.moveaxis(image, -1, 2)
        return image, angle

    def read_angle(self):
        _, left_angles = self.arm_left.get_all_angle()
        left_grasper_angle = self.arm_left.grasper.read_angle()
        _, right_angles = self.arm_right.get_all_angle()
        right_grasper_angle = self.arm_right.grasper.read_angle()
        angles = left_angles + [left_grasper_angle] + right_angles + [right_grasper_angle]
        return angles

    def first(self):
        angles = self.read_angle()
        imgs = self.camera.read_sync()
        self.angles.append(angles)
        self.angles.append(angles)

        img = np.stack([imgs['TOP'], imgs["FRONT"], imgs["LEFT"], imgs["RIGHT"]])
        self.images.append(img)
        self.images.append(img)
        return imgs, angles

    def action(self, action):
        FPS = 2
        bit_width = 2
        for i, step in enumerate(action):
            start_tm = time.time()
            left_angle, left_grasper = step[:6], step[6]
            self.arm_left.puppet.move_to(left_angle, bit_width)
            self.arm_left.grasper.set_angle(left_grasper)
            right_angle, right_grasper = step[7:13], step[13]
            self.arm_right.puppet.move_to(right_angle, bit_width)

            imgs = self.camera.read_sync()
            angles = self.read_angle()
            if i >= 5:
                img = np.stack([imgs['TOP'], imgs["FRONT"], imgs["LEFT"], imgs["RIGHT"]])
                self.images.append(img)
                self.angles.append(angles)

            time_wait(FPS, start_tm)
            bit_width = 1 / (time.time() - start_tm) / 2
            logger.info(f"[{self.step_idx}] bit width {round(bit_width, 4)}")
            self.step_idx += 1


def time_wait(fps: int, tm: float):
    while (time.time() - tm) < (1 / fps):
        time.sleep(0.0001)


def predict(args):
    policy, obs_horizon, action_horizon = build_and_load_policy(args.action_dim, args.ckpt)
    robo = Robo(obs_horizon)
    # robo.start()

    time.sleep(5)
    robo.first()
    while 1:
        obs_images, obs = robo.get_obs()  # (2, 4, 3, 240, 320)  (2, 14)
        tm = time.time()
        action = policy.inference(obs_images, obs)
        logger.info(f"inference time: {round(time.time() - tm, 4)}")
        robo.action(action)
        # (action_horizon, action_dim)

    print('Score: ', max(rewards))

    # visualize
    # from IPython.display import Video
    # vwrite('vis.mp4', imgs)


if __name__ == "__main__":
    device = torch.device('cuda')
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', action='store', type=str, help='ckpt dir', default="diff_ckpt/policy_epoch_best.ckpt")
    parser.add_argument('--data_path', action='store', type=str, help='data path', default='./data/train.zarr')

    # for DIFFUSION
    parser.add_argument('--action_dim', action='store', type=int, help='action dim', default=14)
    parser.add_argument('--obs_horizon', action='store', type=int, help='obs horizon', default=2)

    predict(parser.parse_args())
