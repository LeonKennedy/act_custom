#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: main.py
@time: 2024/7/25 14:52
@desc:
"""
import numpy as np


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats


def get_stats(files: list) -> dict:
    tmp = {
        'agent_pos': np.concatenate([data['qpos'] for data in files]),
        'action': np.concatenate([data['action'] for data in files])
    }
    return {"agent_pos": get_data_stats(tmp['agent_pos']), "action": get_data_stats(tmp['action'])}
