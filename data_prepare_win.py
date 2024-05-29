import pickle
import os
import numpy as np
import random
import matplotlib.pyplot as plt

all_file = []

step = 1

data_path = "./output"

for root, dirs, files in os.walk(os.path.join(data_path, "05_28"), topdown=False):
    for name in files:
        if not name.endswith('.pkl'):
            continue
        filename = os.path.join(root, name)
        all_file.append(filename)
print(all_file, "\nfind", len(all_file), "files")

epsoides = []
max_length = 50
all_show = []
# min_p = np.array([-30, 0, 0, -30, 20, 0])
# max_p = np.array([30, 100, 100, 10, 100, 1])
# min_p = 900
# max_p = 1350
for file in all_file:
    datas = pickle.load(open(file, 'rb'))
    print(file, len(datas))
    epsoide = {'name': file.split("/")[-1].split(".")[0],
               'angle_offset': [],
               'angle_pos': [],
               'gripper': [],
               'angle': [],
               'radians': [],
               'gripper_pos': [],
               'radians_pos': [],
               'image': {
                   'left': [],
                   'front': [],
                   'right': [],
                   'top': []
               },
               'qpos': [],
               'action': []
               }
    if (len(datas)) < max_length:
        continue
    for j in range(len(datas)):
        i = j
        data = datas[i]
        if i < len(datas) - step:
            next_data = datas[i + step]
        else:
            next_data = datas[-1]
        # angle_offset = np.array(next_data['left_puppet'] + next_data['right_puppet']) - np.array(
        #     data['left_puppet'] + data['right_puppet'])
        angle_offset = np.array(next_data['right_puppet']) - np.array(data['right_puppet'])
        # gripper = data['left_gripper']
        right_gripper = data['right_gripper']
        # print(angle_offset, gripper, data['gripper_pos'])
        epsoide['angle_offset'].append(angle_offset)
        # epsoide['qpos'].append(
        #     data['left_puppet'] + [data['left_gripper']] + data['right_puppet'] + [data['right_gripper']])
        # action = next_data['left_puppet'] + [next_data['left_gripper']] + next_data['right_puppet'] + [
        #     next_data['right_gripper']]

        epsoide['qpos'].append(data['right_puppet'] + [data['right_gripper']])
        action = next_data['right_puppet'] + [next_data['right_gripper']]

        epsoide['action'].append(action)
        # epsoide['radians'].append(np.radians(next_robot_status['jointAngle']))
        # epsoide['radians_pos'].append(np.radians(robot_status['jointAngle']))

        # epsoide['gripper'].append([gripper, right_gripper])
        epsoide['gripper'].append([right_gripper])

        # epsoide['gripper_pos'].append([(int(next_data['gripper_pos']) - min_p) / (max_p - min_p)])
        # print(np.radians(next_robot_status['jointAngle']), np.radians(robot_status['jointAngle']))
        # all_show.append(np.radians(next_robot_status['jointAngle']))
        # jointAngle = (np.clip(np.array(robot_status['jointAngle']), min_p, max_p) - min_p) / (max_p - min_p)
        #
        # epsoide['angle_pos'].append(jointAngle)
        filename = os.path.basename(file).split(".")[0]
        basename = os.path.dirname(file)
        image_dir = os.path.join(basename, 'image')
        os.makedirs(image_dir, exist_ok=True)
        for key in ('right', 'top'):
            img_name = os.path.join(image_dir, f"{filename}_{key}_{i}.jpg")
            epsoide["image"][key].append(img_name)
            if not os.path.exists(img_name):
                with open(img_name, 'wb') as f:
                    f.write(data[key])

    epsoide['qpos'] = np.array(epsoide['qpos'], dtype=np.float32)
    epsoide['action'] = np.array(epsoide['action'], dtype=np.float32)

    epsoides.append(epsoide)
    # break

# print(np.array(all_show)[:, 1].shape)
# plt.figure(1)
# plt.hist(np.array(all_show)[:, 1], bins=256, facecolor="blue", edgecolor="black", alpha=0.7)
# plt.xlabel("x")
# plt.ylabel("dis")
# plt.show()
# -30, 30
# 0, 100
# 0, 100
# -25, 10
# 20, 100
# 0
random.shuffle(epsoides)
train_ratio = 0.9
# train = epsoides[:int(len(epsoides) * train_ratio)]
train = epsoides
test = epsoides[int(len(epsoides) * train_ratio):]
pickle.dump(train, open(os.path.join(data_path, 'train_data.pkl'), 'wb'))
pickle.dump(test, open(os.path.join(data_path, 'test.pkl'), 'wb'))
print(len(train), len(test))
