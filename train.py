import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from my_utils import load_data  # data functions
from my_utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from policy import ACTPolicy


def main(args):
    set_seed(args['seed'])
    # command line parameters
    ckpt_dir = args['ckpt_dir']
    task_name = args['task_name']
    ckpt_dir = os.path.join(ckpt_dir, task_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size'] + 2
    num_epochs = args['num_epochs']

    # get task parameters
    from constants import SIM_TASK_CONFIGS
    task_config = SIM_TASK_CONFIGS[task_name]
    dataset_file = task_config['dataset_file']
    camera_names = task_config['camera_names']

    # fixed parameters
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
        'ckpt': args['ckpt'],
        'state_dim': 14,
        'lr': args['lr'],
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'camera_names': camera_names,
    }

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_file, camera_names, batch_size_train,
                                                           batch_size_val, args['chunk_size'])

    # save dataset stats
    os.makedirs(ckpt_dir, exist_ok=True)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    save_data = {"stats": stats, "chunk_size": args['chunk_size']}
    train_bc(train_dataloader, val_dataloader, config, save_data)


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    # print(image_data.shape, qpos_data.shape, action_data.shape, is_pad.shape)
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None


def load_ckpt(policy, ckpt_path):
    print("load state", ckpt_path)
    params = torch.load(ckpt_path)
    print("epoch", params['epoch'], "loss", params['loss'])
    policy.load_state_dict(params['weight'])


def save_check_point(weight, name: str, save_data: dict):
    save_data["weight"] = weight
    torch.save(save_data, name)


def train_bc(train_dataloader, val_dataloader, config, save_data):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = ACTPolicy(policy_config)
    policy.cuda()
    if config['ckpt'] and os.path.exists(config['ckpt']):
        load_ckpt(policy, config['ckpt'])
    optimizer = policy.configure_optimizers()

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = (None, None)
    for epoch in tqdm(range(num_epochs)):
        save_data['epoch'] = epoch
        print(f'\nEpoch {epoch}', "best info:", best_ckpt_info)
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
            save_data["loss"] = epoch_val_loss
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss)
                save_check_point(policy.state_dict(), os.path.join(ckpt_dir, f'policy_best_runtime.ckpt'), save_data)
        summary_string = f'Val   loss:   {epoch_val_loss:.5f} '
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.4f} '
        print(summary_string)

        if epoch > 0 and epoch % 300 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            save_check_point(policy.state_dict(), ckpt_path, save_data)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

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
        summary_string = f'Train loss: {epoch_train_loss:.5f} '
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.4f} '
        print(summary_string)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    save_check_point(policy.state_dict(), ckpt_path, save_data)

    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f}')

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
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--ckpt', action='store', type=str, help='ckpt', default=None, required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=32)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=6000)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=True)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=True)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=2800)

    main(vars(parser.parse_args()))
