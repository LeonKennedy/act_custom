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
from diffusers import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm.auto import tqdm
from loguru import logger

from camera import CameraGroup
from data import EpisodicDataset
from dr import build_two_arm
from dr.utils import fps_wait
from policy_diffusion import build_policy
from task_config import TASK_CONFIG
from action_chunk import ActionChunk


def build_and_load_policy(action_dim, ckpt_path: str):
    if os.path.exists(ckpt_path):
        params = torch.load(ckpt_path)
        obs_horizon = params['obs_horizon']
        action_horizon = 8
        policy = build_policy(obs_horizon, action_dim, 3, params['iter_num'], params['stats'], params['weights'])

        del params['weights']
        print(params)
        return policy, obs_horizon, action_horizon
    raise FileNotFoundError("ckpt not found")


class Robo:
    def __init__(self, obs_horizon: int):
        self.arm_left, self.arm_right = build_two_arm(TASK_CONFIG["Pick_Cube"])
        self.camera = CameraGroup()
        self.angles = collections.deque(maxlen=obs_horizon)
        self.images = collections.deque(maxlen=obs_horizon)
        self.step_idx = 0

    def free_master(self):
        self.arm_left.master.free()
        self.arm_right.master.free()

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
        self.angles.append(angles)
        self.angles.append(angles)
        self.angles.append(angles)

        img = self.camera.read_stack()
        self.images.append(img)
        self.images.append(img)
        self.images.append(img)
        return img, angles

    def action(self, action):
        bit_width = 5
        for i, step in enumerate(action):
            start_tm = time.time()
            left_angle, left_grasper = step[:6], step[6]
            self.arm_left.puppet.move_to(left_angle, bit_width)
            self.arm_left.grasper.set_angle(left_grasper)
            right_angle, right_grasper = step[7:13], step[13]
            self.arm_right.puppet.move_to(right_angle, bit_width)
            self.arm_right.grasper.set_angle(right_grasper)

            img = self.camera.read_stack()
            angles = self.read_angle()
            if i >= 5:
                self.images.append(img)
                self.angles.append(angles)

            fps_wait(10, start_tm)
            bit_width = 1 / (time.time() - start_tm) / 2
            logger.info(f"[{self.step_idx}] bit width {round(bit_width, 4)}")
            self.step_idx += 1


def predict(args):
    policy, obs_horizon, action_horizon = build_and_load_policy(args.action_dim, args.ckpt)
    robo = Robo(obs_horizon)
    robo.free_master()
    # key = input("is move to start?[y/n]")
    # if key == 'y':
    #     robo.start()
    policy.noise_scheduler = DDIMScheduler.from_config(policy.noise_scheduler.config)
    policy.noise_scheduler.set_timesteps(15, device)
    time.sleep(4)
    robo.first()
    while 1:
        obs_images, obs = robo.get_obs()  # (2, 4, 3, 240, 320)  (2, 14)
        tm = time.time()
        action = policy.inference(obs_images, obs, args.action_horizon)
        logger.info(f"inference time: {round(time.time() - tm, 4)}, {action.shape}")
        robo.action(action)
        # (action_horizon, action_dim)


if __name__ == "__main__":
    device = torch.device('cuda')
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', action='store', type=str, help='ckpt dir', default="diff_ckpt/policy_epoch_best.ckpt")

    # for DIFFUSION
    parser.add_argument('--action_dim', action='store', type=int, help='action dim', default=14)
    parser.add_argument('--action_horizon', type=int, default=8)

    predict(parser.parse_args())
