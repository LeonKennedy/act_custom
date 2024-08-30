import pickle
import os
import sys

import cv2
import numpy as np
import random

from constants import SIM_TASK_CONFIGS


def get_all_file(raw_data_path: str):
    all_file = []
    for root, dirs, files in os.walk(raw_data_path, topdown=False):
        for name in files:
            if not name.endswith('.pkl'):
                continue
            filename = os.path.join(root, name)
            all_file.append(filename)
    print(all_file, "\nfind", len(all_file), "files")
    return all_file


def process(all_files: list):
    epsoides = []
    image_dir = os.path.join(raw_data_path, 'images')
    os.makedirs(image_dir, exist_ok=True)
    for file in all_files:
        tmp = pickle.load(open(file, 'rb'))
        datas = tmp["data"]
        epsoide = {'name': file.split("/")[-1].split(".")[0],
                   'image': {
                       'left': [],
                       'front': [],
                       'right': [],
                       'top': []
                   },
                   'qpos': [],
                   'action': [],
                   'task': tmp.get('task', 'tea')
                   }
        if (len(datas)) < 40:
            print("warning: ", file, len(datas), "less 40")
            continue
        print(file)

        step = 2 if "08_21_" in file else 1
        for i in range(0, len(datas), step):
            data = datas[i]
            epsoide['qpos'].append(data['left_puppet'] + data['right_puppet'])
            epsoide['action'].append(data['left_master'] + data['right_master'])

            filename = os.path.basename(file).split(".")[0]
            for key, img in data['camera'].items():
                img_name = os.path.join(image_dir, f"{filename}_{key}_{i}.jpg")
                epsoide["image"][key.lower()].append(img_name)
                if not os.path.exists(img_name):
                    cv2.imwrite(img_name, img)

        epsoide['qpos'] = np.array(epsoide['qpos'], dtype=np.float32)
        epsoide['action'] = np.array(epsoide['action'], dtype=np.float32)
        print(file, "action shape:", epsoide['action'].shape, "qpos shape:", epsoide['qpos'].shape)
        epsoides.append(epsoide)
    return epsoides

    # break


if __name__ == '__main__':
    # task_name = 'multi'
    train_data_path = "output/multi_train_data.pkl"
    test_data_path = "output/multi_test_data.pkl"
    base_path = "output"
    all_data = []
    for task_name in ["cube", "tea"]:
        raw_data_path = os.path.join(base_path, task_name)
        all_files = get_all_file(raw_data_path)
        data = process(all_files)
        all_data.extend(data)

    random.shuffle(all_data)
    train_ratio = 0.9
    train = all_data
    test = all_data[int(len(all_data) * train_ratio):]
    pickle.dump(train, open(train_data_path, 'wb'))
    pickle.dump(test, open(test_data_path, 'wb'))
    print("epsoide num: train", len(train), "test", len(test))
