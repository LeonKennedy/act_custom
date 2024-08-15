import time
from typing import List

import torch
import numpy as np
import os
import pickle
import argparse
from loguru import logger

from camera import CameraGroup
from constants import DT, SIM_TASK_CONFIGS
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from policy import ACTPolicy
from dr import build_two_arm
from dr.utils import fps_wait
from dr.constants import FPS
from task_config import TASK_CONFIG
from action_chunk import ActionChunk


# import IPython
#
# e = IPython.embed


class Robo:
    def __init__(self):
        self.arm_left, self.arm_right = build_two_arm(TASK_CONFIG["Pick_Cube"])
        self.camera = CameraGroup()
        self.step_idx = 0
        self.fps = FPS
        self.bit_width = self.fps / 2

    def free_master(self):
        self.arm_left.master.free()
        self.arm_right.master.free()

    def start(self):
        # free master
        self.arm_left.master.free()
        self.arm_right.master.free()
        self.arm_left.puppet.move_to1([0, -10, -90, -20, 90, 0])
        self.arm_right.puppet.move_to1([0, 0, 90, 0, -86, 0])

    def read_angle(self) -> List:
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

    def action(self, action, s):
        left_angle, left_grasper = action[:6], action[6]
        self.arm_left.puppet.move_to(left_angle, self.bit_width)
        self.arm_left.grasper.set_angle(left_grasper)
        right_angle, right_grasper = action[7:13], action[13]
        self.arm_right.puppet.move_to(right_angle, self.bit_width)
        self.arm_right.grasper.set_angle(right_grasper)

        fps_wait(self.fps, s)
        self.bit_width = 1 / (time.time() - s) / 2
        logger.info(f"[{self.step_idx}] bit width {round(self.bit_width, 4)}")
        self.step_idx += 1


class RoboActionChunk(Robo):
    pass


def main(args):
    set_seed(0)
    # command line parameters
    task_name = args['task_name']

    # get task parameters
    task_config = SIM_TASK_CONFIGS[task_name]
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    policy_config = {'lr': args['lr'],
                     'num_queries': args['chunk_size'],
                     'kl_weight': args['kl_weight'],
                     'hidden_dim': args['hidden_dim'],
                     'dim_feedforward': args['dim_feedforward'],
                     'lr_backbone': lr_backbone,
                     'backbone': backbone,
                     'enc_layers': enc_layers,
                     'dec_layers': dec_layers,
                     'nheads': nheads,
                     'camera_names': camera_names,
                     }

    config = {
        'ckpt': args['ckpt'],
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'camera_names': camera_names,
        'real_robot': True
    }

    success_rate, avg_return = eval_bc(config, save_episode=False)


def eval_bc(config, save_episode=True):
    set_seed(0)
    ckpt = config['ckpt']
    policy_config = config['policy_config']

    # load policy and stats
    policy = ACTPolicy(policy_config)
    params = torch.load(ckpt)
    loading_status = policy.load_state_dict(params['weight'])
    del params['weight']
    print("loading status: ", loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt}', params)
    stats = params['stats']
    chunk_size = params['chunk_size']

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    robo = RoboActionChunk()

    # flag = input("is need move to init?(t)")
    # if flag == 't':
    #     puppet_left.move_to([-125.776, 20.996, -50.522, 8.501, 92.796, -40.213])
    #     puppet_right.move_to([-50.962, -18.231, 47.637, -12.737, -95.001, 24.392])
    robo.free_master()

    time.sleep(2)

    query_frequency = 1
    num_queries = policy_config['num_queries']

    # max_timesteps = 40  # may increase for real-world tasks
    # all_time_actions = np.zeros([max_timesteps, num_queries, 14])

    chunker = ActionChunk(chunk_size)
    t = 0
    while True:
        start = time.time()
        # Observation

        if t % query_frequency == 0:
            all_cam_images = robo.camera.read_stack()
            image_data = torch.from_numpy(all_cam_images)
            image_data = torch.einsum('k h w c -> k c h w', image_data)
            image_data = image_data / 255.0

            qpos = robo.read_angle()
            qpos = np.array(qpos, dtype=np.float32)
            qpos_data = torch.from_numpy(qpos).float()
            qpos_data = pre_process(qpos_data)

            with torch.inference_mode():
                start = time.time()
                all_actions = policy(qpos_data.unsqueeze(0).cuda(), image_data.unsqueeze(0).cuda())
                print(all_actions.shape, '模型预测耗时:', (time.time() - start))

        # ACTION CHUNK

        raw_action = chunker.action_step(all_actions)
        # all_time_actions[:, :-1] = all_time_actions[:, 1:]
        # if t < max_timesteps:
        #     all_time_actions[[t], :num_queries] = all_actions.cpu().numpy()
        # else:
        #     all_time_actions[:-1] = all_time_actions[1:]
        #     all_time_actions[max_timesteps - 1, :num_queries] = all_actions.cpu().numpy()
        #
        # actions_for_curr_step = all_time_actions[:, 0]
        # actions_populated = np.all(actions_for_curr_step != 0, axis=1)
        # actions_for_curr_step = actions_for_curr_step[actions_populated]
        # k = 0.01
        # exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        # exp_weights = exp_weights / exp_weights.sum()
        # exp_weights = exp_weights[:, np.newaxis]
        # raw_action = (actions_for_curr_step * exp_weights).sum(axis=0)
        action = post_process(raw_action)

        ##  DOING ROBOTS
        # action = action + np.random.normal(loc=0.0, scale=0.05, size=len(action))
        print('当前指令:', action)
        robo.action(action, start)

        t += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default="test_grap")
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False, default=10)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False, default=100)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False, default=512)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False,
                        default=3200)

    main(vars(parser.parse_args()))
