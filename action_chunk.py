#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: action_chunk.py
@time: 2024/8/14 15:45
@desc:
"""
import numpy as np


class ActionChunk:
    def __init__(self, chunk_size, state_dim: int = 14):
        self.chunk_size = chunk_size
        self.max_time_steps = 40
        self.num_queries = chunk_size
        self.t = 0
        self.all_time_actions = np.zeros([self.max_time_steps, self.num_queries, state_dim])

    def step(self):
        self.all_time_actions[:, :-1] = self.all_time_actions[:, 1:]
        if self.t >= self.all_time_actions.shape[0]:
            self.all_time_actions[:-1] = self.all_time_actions[1:]

        self.t += 1
        return self.get_action()

    def action_step(self, action):
        self.all_time_actions[:, :-1] = self.all_time_actions[:, 1:]
        if self.t < self.all_time_actions.shape[0]:
            self.all_time_actions[self.t] = action.cpu().numpy()[0]
        else:
            self.all_time_actions[:-1] = self.all_time_actions[1:]
            self.all_time_actions[-1] = action.cpu().numpy()[0]
        self.t += 1
        return self.get_action()

    def _weight_sum(self, action_step):
        actions_populated = np.all(action_step != 0, axis=1)
        actions_for_curr_step = action_step[actions_populated]
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = exp_weights[:, np.newaxis]
        raw_action = (actions_for_curr_step * exp_weights).sum(axis=0)
        return raw_action

    def get_action(self):
        actions_for_curr_step = self.all_time_actions[:, 0]
        return self._weight_sum(actions_for_curr_step)
