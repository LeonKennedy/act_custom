#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: diffusion.py
@time: 2024/8/22 16:59
@desc:
"""

import pickle
import cv2
import torch
import zarr
import os
import numpy as np


def create_sample_indices(episode_ends: np.ndarray, sequence_length: int, pad_before: int = 0, pad_after: int = 0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


class EpisodicDataset(torch.utils.data.Dataset):

    # episode_ids, dataset_dir, camera_names, norm_stats, chunk_size
    def __init__(self, path: str, obs_horizon: int = 2, pred_horizon: int = 16):
        super(EpisodicDataset).__init__()
        dataset_root = zarr.open(path, 'r')
        self.dataset_root = dataset_root

        # # float32, [0,1], (N, 4, 360, 640, 3)
        # self.cache_length = 30000
        # self.cache_img = dataset_root['data']['img'][:]
        # self.cache_img = np.moveaxis(self.cache_img, -1, 2)
        # # (N, 4, 3, 360, 640)

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'agent_pos': dataset_root['data']['state'][:],
            'action': dataset_root['data']['action'][:]
        }
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon + obs_horizon,
            pad_before=0,
            pad_after=0)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        # normalized_train_data['image'] = train_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )
        img = self.dataset_root["data"]["img"][buffer_start_idx: buffer_start_idx + self.obs_horizon]
        nsample['image'] = np.moveaxis(img, -1, 2)
        nsample['agent_pos'] = nsample['agent_pos'][:self.obs_horizon, :]
        nsample['action'] = nsample['action'][self.obs_horizon:, :]
        return nsample

    # def get_image_half_cache(self, start_idx: int, step: int):
    #     if start_idx + step >= self.cache_length:
    #         tmp = self.dataset_root["data"]["img"][start_idx: start_idx + step]
    #         return np.moveaxis(tmp, -1, 2)
    #     else:
    #         return self.cache_img[start_idx: start_idx + step]


def build_dataloader(data_path: str, batch_size: int, obs_horizon: int, pred_horizon: int):
    dataset = EpisodicDataset(data_path, obs_horizon, pred_horizon)
    print("data lenght", len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        num_workers=5,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['image'].shape:", batch['image'].shape)
    print("batch['agent_pos'].shape:", batch['agent_pos'].shape)
    print("batch['action'].shape", batch['action'].shape)
    return dataloader, dataset.stats


def _stack_img(images: dict, start_idx: int, step: int, camera_names: list) -> np.ndarray:
    all_img = []
    for i in range(step):
        step_img = []
        for key in camera_names:
            p = images[key][start_idx + i]
            if os.path.exists(p):
                step_img.append(cv2.imread(p))
            else:
                raise FileNotFoundError(p)
        all_img.append(np.stack(step_img))
    return np.stack(all_img)


class EpisodicDataset2(torch.utils.data.Dataset):

    def __init__(self, data: dict, obs_horizon: int, pred_horizon: int, camera_names: list, stats: dict):
        super(EpisodicDataset).__init__()
        self.data = data
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.camera_names = camera_names

        normalized_data = {}
        normalized_data['qpos'] = normalize_data(data['qpos'], stats['agent_pos'])
        normalized_data['action'] = normalize_data(data['action'], stats['action'])
        self.normalized_data = normalized_data

    def __len__(self) -> int:
        return self.data['qpos'].shape[0] - self.obs_horizon - self.pred_horizon

    def __getitem__(self, idx):
        imgs = _stack_img(self.data['image'], idx, self.obs_horizon, self.camera_names)
        agent_pos = self.normalized_data['qpos'][idx: idx + self.obs_horizon]
        action = self.normalized_data['action'][idx + self.obs_horizon: idx + self.obs_horizon + self.pred_horizon]
        return imgs, agent_pos, action





def build_datasets(path, obs_horizon: int, pred_horizon: int, camera_names):
    files = pickle.load(open(path, 'rb'))
    all_ds = []
    stats = get_stats(files)
    for data in files:
        ds = EpisodicDataset2(data, obs_horizon, pred_horizon, camera_names, stats)
        all_ds.append(ds)
    return all_ds, stats


def collate_fn(x):
    image = np.stack([x[i][0] for i in range(len(x))])
    image = np.moveaxis(image, -1, 2) / 255
    pos = np.stack([x[i][1] for i in range(len(x))], dtype=np.float32)
    action = np.stack([x[i][2] for i in range(len(x))], dtype=np.float32)
    return {
        "image": torch.from_numpy(image),
        "agent_pos": torch.from_numpy(pos),
        "action": torch.from_numpy(action)
    }


def build_dataloader2(path: str, batch_size: int, obs_horizon: int, pred_horizon: int):
    camera_names = ['top', 'left', 'right']
    all_ds, stats = build_datasets(path, obs_horizon, pred_horizon, camera_names)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(all_ds),
        batch_size=batch_size,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        num_workers=10,
        # # don't kill worker process afte each epoch
        persistent_workers=True,
        collate_fn=collate_fn
    )
    return dataloader, stats


if __name__ == '__main__':
    data_path = "train.zarr"
    # dl = build_dataloader(data_path, 8, 2, 16)
    # print(ds.stats)
    # dl = build_dataloader2("../output/train_data.pkl", 8, 2, 16)
    dl = build_dataloader3("../output/tea/tea_train_data.pkl", "../output/tea/tea_test_data.pkl", 8, 1, 100)
    for nbatch in dl:
        print(nbatch)
