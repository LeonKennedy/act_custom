#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: act.py
@time: 2024/8/22 16:58
@desc:
"""
import os
import pickle

import cv2
import torch
import numpy as np

from .main import normalize_data, get_stats
from prompt import TextEmbeddingTransformer


def _stack_img(images: dict, idx: int, camera_names: list) -> np.ndarray:
    imgs = []
    for name in camera_names:
        p = images[name][idx]
        if os.path.exists(p):
            imgs.append(cv2.imread(p))
        else:
            raise FileNotFoundError(p)
    return np.stack(imgs)


class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(self, data: dict, pred_horizon: int, camera_names: list, stats: dict):
        super(EpisodicDataset).__init__()
        self.data = data
        self.pred_horizon = pred_horizon
        self.camera_names = camera_names

        normalized_data = {}
        normalized_data['qpos'] = normalize_data(data['qpos'], stats['agent_pos'])
        normalized_data['action'] = normalize_data(data['action'], stats['action'])
        self.normalized_data = normalized_data
        self.emb = TextEmbeddingTransformer()

    def __len__(self) -> int:
        return self.data['qpos'].shape[0] - 20

    def __getitem__(self, idx):
        imgs = _stack_img(self.data['image'], idx, self.camera_names)
        agent_pos = self.normalized_data['qpos'][idx]
        episode_len = self.normalized_data['action'].shape[0]
        is_pad = np.zeros(self.pred_horizon)
        if idx + self.pred_horizon > episode_len:
            padding_action = np.zeros((self.pred_horizon, self.normalized_data['action'].shape[1]), dtype=np.float32)
            actual_len = episode_len - idx
            padding_action[:actual_len] = self.normalized_data['action'][idx:]
            is_pad[actual_len:] = 1
        else:
            padding_action = self.normalized_data['action'][idx: idx + self.pred_horizon]

        prompt_embedding = self.emb.embedding(self.data['task'])
        return imgs, agent_pos, prompt_embedding, padding_action, is_pad


def build_datasets(path, pred_horizon: int, camera_names):
    files = pickle.load(open(path, 'rb'))
    all_ds = []
    stats = get_stats(files)
    for data in files:
        ds = EpisodicDataset(data, pred_horizon, camera_names, stats)
        all_ds.append(ds)
    return all_ds, stats


def collate_fn(x):
    image = np.stack([x[i][0] for i in range(len(x))])
    image = np.moveaxis(image, -1, 2) / 255
    image = image.astype(np.float32)
    pos = np.stack([x[i][1] for i in range(len(x))], dtype=np.float32)
    prompt = np.stack([x[i][2] for i in range(len(x))], dtype=np.float32)
    action = np.stack([x[i][3] for i in range(len(x))], dtype=np.float32)
    is_pad = np.stack([x[i][4] for i in range(len(x))])
    return {
        "image": torch.from_numpy(image),
        "agent_pos": torch.from_numpy(pos),
        "action": torch.from_numpy(action),
        "prompt": torch.from_numpy(prompt),
        "is_pad": torch.from_numpy(is_pad).bool()
    }


def build_dataloader3(path: str, test_path: str, batch_size: int, pred_horizon: int,
                      camera_names: list):
    train_ds, stats = build_datasets(path, pred_horizon, camera_names)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # num_workers=10,
        # # don't kill worker process afte each epoch
        # persistent_workers=True,
        collate_fn=collate_fn
    )
    val_ds, _ = build_datasets(test_path, pred_horizon, camera_names)
    val_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(val_ds),
        batch_size=batch_size,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        num_workers=10,
        # # don't kill worker process afte each epoch
        persistent_workers=True,
        collate_fn=collate_fn
    )
    return dataloader, val_dataloader, stats
