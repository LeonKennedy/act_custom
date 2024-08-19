#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: diffpolicy.py
@time: 2024/7/26 10:04
@desc:
"""
import time
import math
from typing import Union, Callable, Dict

import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
import torchvision
from torch import nn
import torch
import concurrent.futures
from loguru import logger
from data import normalize_data, unnormalize_data


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
                 input_dim,
                 global_cond_dim,
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 kernel_size=5,
                 n_groups=8
                 ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x


def get_resnet(name: str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
        root_module: nn.Module,
        features_per_group: int = 16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def image_embedding_sync(vision_encodes, nimage):
    features = []
    for i in range(nimage.shape[2]):
        # (obs_horizon,512)
        image_feature = vision_encodes[i](nimage[:, :, i].flatten(end_dim=1))
        image_feature = image_feature.reshape(*nimage.shape[:2], -1)  # (B,obs_horizon,512)
        features.append(image_feature)
    image_features = torch.cat(features, dim=-1)  # (B ,obs_horizon, 2048)
    return image_features


def image_embedding_async(vision_encodes, nimage):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exc:
        futures = {}


class DiffusionPolicy:
    def __init__(self, nets: nn.ModuleDict, num_diffusion_iters: int, stats: Dict):
        self.device = torch.device('cuda')
        nets.to(self.device)
        self.noise_scheduler = get_noise_ddpm_schedule(num_diffusion_iters)
        self.num_diffusion_iters = num_diffusion_iters
        self.nets = nets
        self.stats = stats
        self.pred_horizon = 16

    def create_ema(self):
        self.ema = EMAModel(parameters=self.nets.parameters(), power=0.75)

    def forward(self, nimage, nagent_pos, naction):
        # encoder vision features (B, obs, 512 *3)
        image_features = image_embedding_sync(self.nets['vision_encoders'], nimage / 255)

        # concatenate vision feature and low-dim obs # (B, obs, 1536 + 14)
        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

        noise = torch.randn(naction.shape, device=self.device)

        B = nagent_pos.shape[0]  # (B, 16, 14)
        # sample a diffusion iteration for each data point
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)

        # predict the noise residual
        noise_pred = self.nets['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
        return noise_pred, noise

    def ema_step(self):
        # update Exponential Moving Average of the model weights
        self.ema.step(self.nets.parameters())

    def save(self, path: str, save_data: dict):
        save_data["weights"] = self.nets.state_dict()
        save_data["iter_num"] = self.num_diffusion_iters
        save_data["stats"] = self.stats
        torch.save(save_data, path)
        print("save to", path)
        save_data["weights"] = self.ema.state_dict()
        ema_path = path.replace("epoch", "ema")
        torch.save(save_data, ema_path)
        print("save to", ema_path)

    ######     INFERENCE    ######
    def inference(self, nimage, nagent_pos, action_horizon: int = 8) -> np.ndarray:
        start_tm = time.time()
        obs_horizon = nagent_pos.shape[0]
        nagent_poses = normalize_data(nagent_pos, self.stats['agent_pos'])

        # device transfer
        nimage = torch.from_numpy(nimage).to(self.device, dtype=torch.float32)
        nimage = torch.unsqueeze(nimage, dim=0)
        # (2,3,96,96)
        nagent_poses = torch.from_numpy(nagent_poses).to(self.device, dtype=torch.float32)
        nagent_poses = torch.unsqueeze(nagent_poses, dim=0)
        # (2,2)
        tm1 = time.time()
        # infer action
        with torch.no_grad():
            # get image features (B , obs, 512 * 3)
            image_features = image_embedding_sync(self.nets['vision_encoders'], nimage / 255)
            tm2 = time.time()

            # concat with low-dim observations (B, obs, 1536 + 14)
            obs_features = torch.cat([image_features, nagent_poses], dim=-1)

            # reshape observation to (B, 1550 * obs)
            obs_cond = obs_features.flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn((1, self.pred_horizon, nagent_poses.shape[2]), device=self.device)
            naction = noisy_action

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.nets['noise_pred_net'](
                    sample=naction,  # (B, pred_horizon, 14)
                    timestep=k,
                    global_cond=obs_cond
                )  # (B, 16, action_dim)

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample  # (B, 16, action_dim)

            tm3 = time.time()

        # unnormalize action (B, pred_horizon, action_dim)
        naction = naction.detach().to('cpu').numpy()[0]  # (16, 14)
        action_pred = unnormalize_data(naction, self.stats['action'])

        # only take action_horizon number of actions
        # start = obs_horizon - 1
        # end = start + action_horizon
        # action = action_pred[start:end, :]
        logger.info(
            f"time cost: {round(tm1 - start_tm, 4)}, vision encode {round(tm2 - tm1, 4)} noise {round(tm3 - tm2, 4)} end {round(time.time() - tm3), 4}")
        return action_pred


def build_policy(obs_horizon: int, action_dim: int, camera_cnt: int, iter_num: int, stats: Dict,
                 weight=None) -> DiffusionPolicy:
    nets = build_net(obs_horizon, action_dim, camera_cnt)
    if weight is not None:
        nets.load_state_dict(weight)
    return DiffusionPolicy(nets, iter_num, stats)


def build_net(obs_horizon: int, action_dim: int, camera_cnt: int) -> nn.ModuleDict:
    vision_feature_dim = 512
    obs_dim = vision_feature_dim * camera_cnt + action_dim

    vision_encoders = nn.ModuleList([replace_bn_with_gn(get_resnet('resnet18')) for _ in range(camera_cnt)])

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    )

    nets = nn.ModuleDict({
        'vision_encoders': vision_encoders,
        'noise_pred_net': noise_pred_net
    })
    return nets


def get_noise_ddpm_schedule(num_diffusion_iters: int = 100) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )


def get_noise_ddim_schedule(num_diffusion_iters) -> DDIMScheduler:
    ddpm_scheduler = get_noise_ddpm_schedule(num_diffusion_iters)
    return DDIMScheduler.from_config(ddpm_scheduler.config)


def test_build_net(obs_horizon, action_dim, camera_cnt: int = 4):
    pred_horizon = 16
    nets = build_net(obs_horizon, action_dim, camera_cnt)
    with torch.no_grad():
        # example inputs
        images = torch.zeros((1, obs_horizon, camera_cnt, 3, 360, 640))
        agent_pos = torch.zeros((1, obs_horizon, action_dim))
        # vision encoder
        features = []
        for i in range(images.shape[2]):
            image_feature = nets['vision_encoders'][i](images[:, :, i].flatten(end_dim=1))  # (2,512)
            image_feature = image_feature.reshape(*images.shape[:2], -1)  # (1,2,512)
            features.append(image_feature)
        image_features = torch.cat(features, dim=-1)  # (1,2,512 * 4)
        obs = torch.cat([image_features, agent_pos], dim=-1)  # (1, 2, 2048 + 14)

        noised_action = torch.randn((1, pred_horizon, action_dim))
        diffusion_iter = torch.zeros((1,))

        # the noise prediction network
        # takes noisy action, diffusion iteration and observation as input
        # predicts the noise added to action
        noise = nets['noise_pred_net'](
            sample=noised_action,
            timestep=diffusion_iter,
            global_cond=obs.flatten(start_dim=1))  # (1, 1052)

        # illustration of removing noise
        # the actual noise removal is performed by NoiseScheduler
        # and is dependent on the diffusion noise schedule
        denoised_action = noised_action - noise
        print(denoised_action.shape)  # (B, 16, 14)


if __name__ == '__main__':
    # action_horizon = 8
    #
    test_build_net(obs_horizon=2, action_dim=14)
