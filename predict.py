import time

import serial
import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from einops import rearrange

import cv2
from scipy.ndimage import shift

from camera import CameraGroup
from constants import DT, SIM_TASK_CONFIGS
from dr.DrRobot import build_arm, build_puppet
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
# from my_utils import get_angle_all
from policy import ACTPolicy
from dr.constants import GRASPER_NAME, COM_NAME, BAUDRATE, FPS, IMAGE_H, IMAGE_W, CAMERA_TOP, CAMERA_RIGHT
from visualize_episodes import save_videos
from PIL import Image
# from dr import DrEmpower_can, PuppetRight, GRASPER_NAME

import IPython

e = IPython.embed


def main(args):
    set_seed(1)
    # command line parameters
    ckpt_dir = args['ckpt_dir']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    num_epochs = args['num_epochs']

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
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'camera_names': camera_names,
        'real_robot': True
    }
    ckpt_name = "policy_best_runtime.ckpt"
    # ckpt_name = "policy_epoch_4200_seed_0.ckpt"

    success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=False)


def make_policy(policy_config):
    policy = ACTPolicy(policy_config)
    return policy


def make_optimizer(policy):
    optimizer = policy.configure_optimizers()
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def init_camera(camera_id):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)  # top
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(3, IMAGE_W)
    cap.set(4, IMAGE_H)
    return cap


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(0)
    ckpt_dir = config['ckpt_dir']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']

    # load policy and stats
    policy = make_policy(policy_config)
    loading_status = policy.load_state_dict(torch.load(os.path.join(ckpt_dir, ckpt_name)))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_name}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    camera = CameraGroup()
    dr = DrEmpower_can(com=COM_NAME, uart_baudrate=BAUDRATE)
    puppet_left, puppet_right = build_puppet(dr)

    flag = input("is need move to init?(t)")
    if flag == 't':
        puppet_left.move_to([-125.776, 20.996, -50.522, 8.501, 92.796, -40.213])
        puppet_right.move_to([-50.962, -18.231, 47.637, -12.737, -95.001, 24.392])

    time.sleep(2)

    # clean flush
    for i in range(3):
        imgs = camera.read_sync()

    query_frequency = 1
    num_queries = policy_config['num_queries']

    max_timesteps = 40  # may increase for real-world tasks
    all_time_actions = np.zeros([max_timesteps, num_queries, 14])

    t = 0
    while True:
        # Observation
        start = time.time()

        if t % query_frequency == 0:
            images = camera.read_sync()
            all_cam_images = np.stack([
                images["TOP"],
                images["FRONT"],
                images["LEFT"],
                images["RIGHT"]
            ], axis=0)
            image_data = torch.from_numpy(all_cam_images)
            image_data = torch.einsum('k h w c -> k c h w', image_data)
            image_data = image_data / 255.0

            lp, rp, left_puppet_gripper, right_puppet_gripper = get_angle_all(dr)
            qpos = np.array(lp + [left_puppet_gripper] + rp + [right_puppet_gripper]).astype(np.float32)
            qpos_data = torch.from_numpy(qpos).float()
            qpos_data = pre_process(qpos_data)

            with torch.inference_mode():
                start = time.time()
                all_actions = policy(qpos_data.unsqueeze(0).cuda(), image_data.unsqueeze(0).cuda())
                print('模型预测耗时:', (time.time() - start))

        # ACTION CHUNK

        all_time_actions[:, :-1] = all_time_actions[:, 1:]
        if t < max_timesteps:
            all_time_actions[[t], :num_queries] = all_actions.cpu().numpy()
        else:
            all_time_actions[:-1] = all_time_actions[1:]
            all_time_actions[max_timesteps-1, :num_queries] = all_actions.cpu().numpy()

        actions_for_curr_step = all_time_actions[:, 0]
        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = exp_weights[:, np.newaxis]
        # exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1)
        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0)
        # raw_action = raw_action.squeeze(0).cpu().numpy()
        action = post_process(raw_action)

        ##  DOING ROBOTS
        # action = action + np.random.normal(loc=0.0, scale=0.05, size=len(action))
        print('当前指令:', action)
        left_target = action[:6]
        left_gripper_angle = action[6]
        right_target = action[7:13]
        right_gripper_angle = action[13]

        print('目标位置:', left_target, right_target)

        while (time.time() - start) < (1 / FPS):  # t/n=10, sleep 10毫秒
            time.sleep(0.0001)
        a = time.time() - start
        bit_width = 1 / a / 2
        puppet_left.move_to2(left_target, bit_width)
        puppet_left.set_gripper(left_gripper_angle)
        puppet_right.move_to2(right_target, bit_width)
        puppet_right.set_gripper(right_gripper_angle)

        t += 1
        if onscreen_render:
            pass

        # if save_episode:
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    return success_rate, avg_return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default="test_grap")
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False, default=10)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False, default=100)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False, default=512)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False,
                        default=3200)

    main(vars(parser.parse_args()))
