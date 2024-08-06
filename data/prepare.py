# !/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: prepare.py
@time: 2024/7/25 15:47
@desc:
"""
import glob
import os
import pickle
import sys
from itertools import accumulate
from typing import Tuple

import zarr
import numpy as np


def _camera(data):
    imgs = []
    for e in data:
        c = e['camera']
        img = np.stack([c['TOP'], c['LEFT'], c['RIGHT']])  # 4 360, 640, 3
        imgs.append(img)
    return np.stack(imgs)


def _stack(data, key: str):
    return np.stack([e[key] for e in data])


def handle_one(filename: str):
    data = pickle.loads(open(filename, 'rb').read())
    left_master = _stack(data, 'left_master')
    right_master = _stack(data, 'right_master')
    master = np.concatenate([left_master, right_master], axis=1)

    left_puppet = _stack(data, 'left_puppet')
    right_puppet = _stack(data, 'right_puppet')
    puppet = np.concatenate([left_puppet, right_puppet], axis=1)

    image = _camera(data)
    return master, puppet, image


def get_info(filename: str) -> Tuple[int, int, int]:
    data = pickle.loads(open(filename, 'rb').read())
    episode = data[0]
    state_dim = len(episode['left_puppet']) + len(episode['right_puppet'])
    action_dim = len(episode['left_master']) + len(episode['right_master'])
    camera_cnt = len(episode['camera'])
    return state_dim, action_dim, camera_cnt


def main(path: str):
    all_file = glob.glob(os.path.join(path, "*.pkl"))
    assert len(all_file) > 0
    state_dim, action_dim, camera_cnt = get_info(all_file[0])


    root = zarr.open("train.zarr", mode="w")
    # zip_store = zarr.ZipStore("train.zip", mode='w')
    # root = zarr.group(store=zip_store)
    data = root.create_group("data")
    action = data.create_dataset("action", shape=(0, action_dim), chunks=(2000, -1), dtype=np.float32)
    agent_pos = data.create_dataset("state", shape=(0, state_dim), chunks=(2000, -1), dtype=np.float32)
    image = data.create_dataset("img", shape=(0, camera_cnt, 240, 320, 3), chunks=(200, -1, -1, -1, -1), dtype=np.uint8)
    all_file = glob.glob(os.path.join(path, "*.pkl"))
    episode_ends = []
    for f in all_file:
        print(f, "handle")
        master, puppet, image_data = handle_one(f)
        print(f, len(master))

        # use next puppet for action
        master = puppet[1:]
        puppet = puppet[:-1]

        image.append(image_data)
        action.append(master)
        agent_pos.append(puppet)
        episode_ends.append(len(master))

    meta = root.create_group('meta')
    meta.create_dataset("episode_ends", data=list(accumulate(episode_ends)))

    assert meta['episode_ends'][-1] == action.shape[0]
    print(image.info)
    print(action.info)
    print(agent_pos.info)
    print(list(meta.episode_ends))
    return


if __name__ == '__main__':
    data_path = sys.argv[1]
    main(data_path)
