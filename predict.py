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

from constants import DT, SIM_TASK_CONFIGS
from utils import sample_box_pose, sample_insertion_pose  # robot functions
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from policy import ACTPolicy
from dr.constants import GRASPER_NAME, COM_NAME, BAUDRATE, FPS, IMAGE_H, IMAGE_W, CAMERA_TOP, CAMERA_RIGHT
from visualize_episodes import save_videos
from PIL import Image
from dr import DrEmpower_can, PuppetRight, Grasper

import IPython

e = IPython.embed


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
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

    ckpt_name = 'policy_epoch_4800_seed_0.ckpt'
    success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=False)

    print()
    exit()


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
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    task_name = config['task_name']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    cap_top = init_camera(CAMERA_TOP)
    cap_right = init_camera(CAMERA_RIGHT)

    # clean flush
    for i in range(5):
        ret, image = cap_top.read()
        assert ret
        ret, image2 = cap_right.read()
        assert ret

    query_frequency = 1
    num_queries = policy_config['num_queries']

    max_timesteps = 60 # may increase for real-world tasks
    all_time_actions = np.zeros([max_timesteps, max_timesteps + num_queries, 7])


    dr = DrEmpower_can(com=COM_NAME, uart_baudrate=BAUDRATE)
    ser_port = serial.Serial(GRASPER_NAME, BAUDRATE)
    right_puppet = PuppetRight(dr, Grasper(ser_port, 1))
    print('wait 5 section... then move to begin...')
    time.sleep(5)

    t = 0
    while True:
        # Observation
        ret, image_top = cap_top.read()
        ret, image_right = cap_right.read()
        all_cam_images = np.stack([
            cv2.cvtColor(image_top, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
        ], axis=0)
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0

        angles = dr.get_angle_speed_torque_all([i for i in range(1, 13)])
        angles = [row[0] for row in angles]
        right_angles = angles[6:]
        qpos = np.array(right_angles + [right_puppet.gripper_status]).astype(np.float32)
        qpos_data = torch.from_numpy(qpos).float()
        qpos_data = pre_process(qpos_data)

        start = time.time()
        while (time.time() - start) < (1 / FPS):  # t/n=10, sleep 10毫秒
            time.sleep(0.0001)
        a = time.time() - start
        bit_width = 1 / a / 2

        with torch.inference_mode():
            start = time.time()
            all_actions = policy(qpos_data.unsqueeze(0).cuda(), image_data.unsqueeze(0).cuda())
            print('模型预测耗时:', (time.time() - start))

        # ACTION CHUNK

        all_time_actions[:, :-1] = all_time_actions[:, 1:]
        if t == 50:
            all_time_actions[:-1] = all_time_actions[1:]
        all_time_actions[[t], :num_queries] = all_actions.cpu().numpy()
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
        # now = post_process(all_actions[0][0].cpu())
        print('当前指令:', action)
        right_target = action[:6]
        right_gripper = action[-1]
        print('目标位置:', right_target, round(right_gripper))
        # robotPuppet.move_to(action[:6], False)
        # leftPuppet.move_to2(left_target, bit_width)
        # leftPuppet.set_gripper(round(left_gripper))
        right_puppet.move_to2(right_target, bit_width)
        right_puppet.set_gripper(round(right_gripper))

        if t < 50:
            t += 1
        if onscreen_render:
            pass


        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        # with torch.inference_mode():
        #     for t in range(max_timesteps):
        #         ### update onscreen render and wait for DT
        #         if onscreen_render:
        #             pass
        #
        #         ### process previous timestep to get qpos and image_list
        #         obs = ts.observation
        #         if 'images' in obs:
        #             image_list.append(obs['images'])
        #         else:
        #             image_list.append({'main': obs['image']})
        #         qpos_numpy = np.array(obs['qpos'])
        #         qpos = pre_process(qpos_numpy)
        #         qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        #         qpos_history[:, t] = qpos
        #         curr_image = get_image(ts, camera_names)
        #
        #         ### query policy
        #         if t % query_frequency == 0:
        #             all_actions = policy(qpos, curr_image)
        #         if temporal_agg:
        #             all_time_actions[[t], t:t + num_queries] = all_actions
        #             actions_for_curr_step = all_time_actions[:, t]
        #             actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
        #             actions_for_curr_step = actions_for_curr_step[actions_populated]
        #             k = 0.01
        #             exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        #             exp_weights = exp_weights / exp_weights.sum()
        #             exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
        #             raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        #         else:
        #             raw_action = all_actions[:, t % query_frequency]
        #
        #         ### post-process actions
        #         raw_action = raw_action.squeeze(0).cpu().numpy()
        #         action = post_process(raw_action)
        #         target_qpos = action
        #
        #         ### for visualization
        #         qpos_list.append(qpos_numpy)
        #         target_qpos_list.append(target_qpos)

        if real_robot:
            pass

        # if save_episode:
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    return success_rate, avg_return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', default="ckpt2")
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
