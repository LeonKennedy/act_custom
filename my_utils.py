import pickle
import cv2

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import IPython

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, chunk_size):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = True
        self.files = pickle.load(open(dataset_dir, 'rb'))
        self.chunk_size = chunk_size
        self.fixed_length = chunk_size + 10
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        file = self.files[episode_id]
        original_action_shape = file['action'].shape
        episode_len = original_action_shape[0]
        start_ts = np.random.choice(episode_len - self.chunk_size + 1)
        # get observation at start_ts only
        qpos = file['qpos'][start_ts]
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = file["image"][cam_name][start_ts]
        # get all actions after and including start_ts

        action_cur_ts = max(0, start_ts - 1)
        if action_cur_ts + self.fixed_length > episode_len:
            padded_action = np.zeros((self.fixed_length, original_action_shape[1]), dtype=np.float32)
            action = file['action'][action_cur_ts:]  # hack, to make timesteps more aligned
            action_len = episode_len - action_cur_ts  # hack, to make timesteps more aligned

            padded_action[:action_len] = action
            is_pad = np.zeros(self.fixed_length)
            is_pad[action_len:] = 1
        else:
            padded_action = file['action'][action_cur_ts: action_cur_ts + self.fixed_length]
            is_pad = np.zeros(self.fixed_length)

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            img = cv2.imread(image_dict[cam_name])
            all_cam_images.append(img)
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir):
    all_qpos_data = []
    all_action_data = []
    files = pickle.load(open(dataset_dir, 'rb'))
    for file in files:
        qpos = np.array(file['qpos'])
        action = np.array(file['action'])
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    all_qpos_data = torch.concatenate(all_qpos_data)
    all_action_data = torch.concatenate(all_action_data)

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    # normalize action data
    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = all_action_data.std(dim=[0], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, camera_names, batch_size_train, batch_size_val, chunk_size):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    files = pickle.load(open(dataset_dir, 'rb'))
    num_episodes = len(files)
    train_ratio = 0.9
    shuffled_indices = np.random.permutation(num_episodes)
    # train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    train_indices = shuffled_indices
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    print("train data length:", len(train_indices), "val data length:", len(val_indices))

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, chunk_size)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, chunk_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val+2, pin_memory=True)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# def get_angle_all(dr):
#     while 1:
#         angles = dr.get_angle_speed_torque_all([i for i in range(1, 27)])
#         if angles:
#             angles = [row[0] for row in angles]
#             return angles[12:18], angles[18:24], angles[24], angles[:6], angles[6:12], angles[25]
#         print("read again!!!!!!!!!!!!!!!!!!!!!!!")




if __name__ == '__main__':
    local_data = "output/train_data.pkl"
    files = pickle.load(open(local_data, 'rb'))
    num_episodes = len(files)
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(0.9 * num_episodes)]
    val_indices = shuffled_indices[int(0.9 * num_episodes):]

    norm_stats = get_norm_stats(local_data)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, local_data, ["top", "right"], norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True,
                                  num_workers=8, prefetch_factor=1)
    for x in train_dataloader:
        print(x)
