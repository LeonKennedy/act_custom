import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import IPython
from PIL import Image

e = IPython.embed

MAX_LENGTH = 50


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.files = pickle.load(open(dataset_dir, 'rb'))
        print(len(episode_ids))
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode

        episode_id = self.episode_ids[index]
        file = self.files[episode_id]
        start_index = np.random.randint(0, len(file['qpos']) - MAX_LENGTH)
        # 获取子数组
        data = {'image': {}}
        data['qpos'] = file['qpos'][start_index:start_index + MAX_LENGTH]
        data['action'] = file['action'][start_index:start_index + MAX_LENGTH]

        # original_action_shape = file.shape
        episode_len = len(data['qpos'])
        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)
        # get observation at start_ts only
        # print(start_index, episode_len, start_ts)
        qpos = data['qpos'][start_ts]
        # qvel = root['/observations/qvel'][start_ts]
        image_dict = dict()
        for cam_name in self.camera_names:
            data['image'][cam_name] = file['image'][cam_name][start_index:start_index + MAX_LENGTH]
            image_dict[cam_name] = data['image'][cam_name][start_ts]
        # get all actions after and including start_ts
        action = data['action'][max(0, start_ts - 1):]  # hack, to make timesteps more aligned
        action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

        padded_action = np.zeros([episode_len, 7], dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            img = Image.open(image_dict[cam_name])
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


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    files = pickle.load(open(dataset_dir, 'rb'))
    for file in files:
        qpos = np.array(file['qpos'])
        # qvel = root['/observations/qvel'][()]
        action = np.array(file['action'])
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.concatenate(all_qpos_data)
    all_action_data = torch.concatenate(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = all_action_data.std(dim=[0], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    files = pickle.load(open(dataset_dir, 'rb'))
    num_episodes = len(files)
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,
                                  num_workers=4, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True)

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


if __name__ == '__main__':
    local_data = "output/train_data.pkl"
    files = pickle.load(open(local_data, 'rb'))
    num_episodes = len(files)
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(0.9 * num_episodes)]
    val_indices = shuffled_indices[int(0.9 * num_episodes):]

    norm_stats = get_norm_stats(local_data, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, local_data, ["top", "right"], norm_stats)

    for x in train_dataset:
        print(x)