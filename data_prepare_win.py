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
    min_length = 40
    step = 1

    for file in all_files:
        datas = pickle.load(open(file, 'rb'))
        epsoide = {'name': file.split("/")[-1].split(".")[0],
                   'image': {
                       'left': [],
                       'front': [],
                       'right': [],
                       'top': []
                   },
                   'qpos': [],
                   'action': []
                   }
        if (len(datas)) < min_length:
            continue
        print(file, len(datas))
        for j in range(len(datas)):
            i = j
            data = datas[i]
            if i < len(datas) - step:
                next_data = datas[i + step]
            else:
                next_data = datas[-1]
            epsoide['qpos'].append(data['left_puppet'] + data['right_puppet'])
            epsoide['action'].append(next_data['left_puppet'] + next_data['right_puppet'])

            filename = os.path.basename(file).split(".")[0]
            image_dir = os.path.join(raw_data_path, 'images')
            os.makedirs(image_dir, exist_ok=True)
            for key, img in data['camera'].items():
                img_name = os.path.join(image_dir, f"{filename}_{key}_{i}.jpg")
                epsoide["image"][key.lower()].append(img_name)
                if not os.path.exists(img_name):
                    cv2.imwrite(img_name, img)

        epsoide['qpos'] = np.array(epsoide['qpos'], dtype=np.float32)
        epsoide['action'] = np.array(epsoide['action'], dtype=np.float32)

        epsoides.append(epsoide)
        # break

    random.shuffle(epsoides)
    train_ratio = 0.9
    train = epsoides
    test = epsoides[int(len(epsoides) * train_ratio):]
    pickle.dump(train, open(train_data_path, 'wb'))
    pickle.dump(test, open(test_data_path, 'wb'))
    print("epsoide num: train", len(train), "test", len(test))


if __name__ == '__main__':
    task_name = sys.argv[1]
    conf = SIM_TASK_CONFIGS[task_name]
    train_data_path = conf['dataset_file']
    test_data_path = conf['test_dataset_file']
    base_path = "output"
    raw_data_path = os.path.join(base_path, task_name)
    if os.path.exists(raw_data_path):
        all_files = get_all_file(raw_data_path)
        process(all_files)
    else:
        raise FileNotFoundError(f"{raw_data_path} not found")
