import pickle
import os

import cv2
import numpy as np
import random


def get_all_file():
    all_file = []

    for root, dirs, files in os.walk(os.path.join(data_path, "06_28"), topdown=False):
        for name in files:
            if not name.endswith('.pkl'):
                continue
            filename = os.path.join(root, name)
            all_file.append(filename)
    print(all_file, "\nfind", len(all_file), "files")
    return all_file


def process():
    epsoides = []
    min_length = 100
    step = 1

    # min_p = np.array([-30, 0, 0, -30, 20, 0])
    # max_p = np.array([30, 100, 100, 10, 100, 1])
    # min_p = 900
    # max_p = 1350
    for file in get_all_file():
        datas = pickle.load(open(file, 'rb'))
        print(file, len(datas))
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
            basename = os.path.dirname(file)
            image_dir = os.path.join(data_path, 'images')
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
    # train = epsoides[:int(len(epsoides) * train_ratio)]
    train = epsoides
    test = epsoides[int(len(epsoides) * train_ratio):]
    pickle.dump(train, open(os.path.join(data_path, 'train_data.pkl'), 'wb'))
    pickle.dump(test, open(os.path.join(data_path, 'test.pkl'), 'wb'))
    print("epsoide num: train", len(train), "test", len(test))


if __name__ == '__main__':
    data_path = "./output"
    process()
