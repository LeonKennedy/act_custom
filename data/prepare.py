#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: prepare.py
@time: 2024/7/25 15:47
@desc:
"""
import os
import glob
import pickle
from itertools import accumulate
import zarr
import numpy as np


def _camera(data):
    imgs = []
    for e in data:
        c = e['camera']
        img = np.stack([c['TOP'], c['FRONT'], c['LEFT'][:360], c['RIGHT'][:360]])  # 4 360, 640, 3
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


def main(path: str):
    L = 1000
    root = zarr.open("train.zarr", mode="w")
    # zip_store = zarr.ZipStore("train.zip", mode='w')
    # root = zarr.group(store=zip_store)
    data = root.create_group("data")
    action = data.create_dataset("action", shape=(0, 14), chunks=(2000, -1), dtype=np.float32)
    agent_pos = data.create_dataset("state", shape=(0, 14), chunks=(2000, -1), dtype=np.float32)
    image = data.create_dataset("img", shape=(0, 4, 360, 640, 3), chunks=(200, -1, -1, -1, -1), dtype=np.uint8)
    all_file = glob.glob(path + "/*.pkl")
    episode_ends = []
    for f in all_file:
        master, puppet, image_data = handle_one(f)
        print(f, len(master))
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
    data_path = "../output/07_24"
    print(os.getcwd())
    main(data_path)
