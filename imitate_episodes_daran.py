import random

import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import serial
from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils_daran import load_data  # data functions
from utils_daran import sample_box_pose, sample_insertion_pose  # robot functions
from utils_daran import compute_dict_mean, set_seed, detach_dict  # helper functions
from policy import ACTPolicy, CNNMLPPolicy
import cv2
import DrEmpower_can as Dr # 忽略此处的报错
from DrRobot import Robot
import time
from PIL import Image
import IPython


e = IPython.embed


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = False
    from constants import SIM_TASK_CONFIGS
    task_config = SIM_TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
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
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone': backbone, 'num_queries': 1,
                         'camera_names': camera_names, }
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train,
                                                           batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image




FPS = 20


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
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

    width = 640
    height = 360
    cap_top = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # top
    cap_top.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap_top.set(3, width)
    cap_top.set(4, height)
    cap_right = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # right
    cap_right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap_right.set(3, width)
    cap_right.set(4, height)

    for i in range(5):
        ret, image = cap_top.read()
        ret, image2 = cap_right.read()

    i = 0
    num_queries = 40
    all_time_actions = torch.zeros([1000, 1000 + num_queries, 7]).cuda()
    t = 0

    dr = Dr.DrEmpower_can(com=ServoName, uart_baudrate=Baudrate)
    gripper = serial.Serial(GripperName, Baudrate)

    rightPuppet = Robot([7, 8, 9, 10, 11, 12], dr, gripper, 1)
    # leftPuppet = Robot([7, 8, 9, 10, 11, 12], dr, gripper, 1)
    rightPuppet.move_to([0, 0, -30, 0, 0, 0])
    # leftPuppet.move_to([25.397, -19.14, 30.52, -23.025, -16.829, 8.837])
    print('move to begin')
    time.sleep(5)

    save_mode = False
    grasped = False

    while True:
        i += 1
        begin = time.time()
        ret, image_top = cap_top.read()
        ret, image_right = cap_right.read()

        cv2.imwrite('./run/run_0_%s.jpg' % i, image_top)
        cv2.imwrite('./run/run_1_%s.jpg' % i, image_right)

        top = Image.open('./run/run_0_%s.jpg' % i)
        right = Image.open('./run/run_1_%s.jpg' % i)

        print(np.array(image).shape)
        all_cam_images = [np.array(top), np.array(right), ]
        all_cam_images = np.stack(all_cam_images, axis=0)
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0
        # min_p = np.array([-20, 0, 0, -30, 50, -50])
        # max_p = np.array([10, 100, 100, 5, 110, 100])
        angles = dr.get_angle_speed_torque_all([i for i in range(1, 13)])
        angles = [row[0] for row in angles]
        right_angles = angles[6:]
        # robot_status = robotPuppet.get_angles()
        # jointAngle = (np.clip(np.array(robot_status['jointAngle']), min_p, max_p) - min_p) / (max_p - min_p)
        qpos = np.array(right_angles + [rightPuppet.gripper_status]).astype(np.float32)
        qpos_data = torch.from_numpy(qpos).float()
        qpos_data = (qpos_data - stats["qpos_mean"]) / stats["qpos_std"]
        start = time.time()
        while (time.time() - start) < (1 / FPS):  # t/n=10, sleep 10毫秒
            time.sleep(0.0001)
        a = time.time() - start
        bit_width = 1 / a / 2
        with torch.inference_mode():
            start = time.time()
            all_actions = policy(qpos_data.unsqueeze(0).cuda(), image_data.unsqueeze(0).cuda())
            print('模型预测耗时:', (time.time() - start))
            if random.random() > 0.93:
                print('重置!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                all_time_actions = torch.zeros([1000, 1000 + num_queries, 7]).cuda()
            if True:
                all_time_actions[[t], t:t + num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action = action + np.random.normal(loc=0.0, scale=0.05, size=len(action))
                now = post_process(all_actions[0][0].cpu())
                print('当前指令:', now)
                # print('平均位移:', action)
                # left_target = action[:6]
                # left_gripper = action[6]
                # right_target = action[7:13]
                # right_gripper = action[-1]
                right_target = action[:6]
                right_gripper = action[-1]

                # action[-1] = 2 if now[-1] > 1.6 else 1
                print('目标位置:', right_target, round(right_gripper))
                # robotPuppet.move_to(action[:6], False)
                # leftPuppet.move_to2(left_target, bit_width)
                # leftPuppet.set_gripper(round(left_gripper))
                rightPuppet.move_to2(right_target, bit_width)
                rightPuppet.set_gripper(round(right_gripper))
            else:
                action = post_process(all_actions[0][0].cpu()).numpy()
                # robot_status = robot.get_robot_status()
                left_target = action[:6]
                left_gripper = action[6]
                right_target = action[7:13]
                right_gripper = action[-1]
                print('目标位置:', left_target, right_target, round(left_gripper), round(right_gripper))
                leftPuppet.move_to2(left_target, bit_width)
                leftPuppet.set_gripper(round(left_gripper))
                rightPuppet.move_to2(right_target, bit_width)
                rightPuppet.set_gripper(round(right_gripper))
            t += 1

            # robot_status = robotPuppet.get_angles()
            # print('本次结束位置:', robot_status)
            while (time.time() - start) < (1 / FPS):  # t/n=10, sleep 10毫秒
                time.sleep(0.0001)
            bit_width = 1 / (time.time() - start) / 2  # 时刻监控在 t>n * bit_time 情况下单条指令发送的时间
            print((time.time() - start), bit_width)


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    # print(image_data.shape, qpos_data.shape, action_data.shape, is_pad.shape)
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None


def load_ckpt(policy, ckpt_path):
    print("load state", ckpt_path)
    a = torch.load(ckpt_path)
    policy.load_state_dict(a)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    # load_ckpt(policy, "train/policy_epoch_16100_seed_0.ckpt")
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs - 1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    ServoName = "COM5"
    GripperName = "COM4"
    Baudrate = 115200  # 串口波特率，与CAN模块的串口波特率一致，（出厂默认为 115200，最高460800）
    main(vars(parser.parse_args()))
