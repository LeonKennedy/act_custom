#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: record.py
@time: 2024/5/20 15:52
@desc:
"""
import os
import pickle
import time
from datetime import datetime
import concurrent.futures

from dr import build_two_arm, Arm
from dr.constants import FPS, BUTTON_NAME
from button import Button
from camera import CameraGroup
from task_config import TASK_CONFIG


class Recorder:

    def __init__(self, arm_left: Arm, arm_right: Arm):
        self.folder_name = "%s" % (datetime.now().strftime("%m_%d"))
        os.makedirs("output/%s" % self.folder_name, exist_ok=True)
        self.arm_left = arm_left
        self.arm_right = arm_right
        self.camera = CameraGroup()

    def clear_uart(self):
        self.arm_left.clear_uart()
        self.arm_right.clear_uart()

    def record(self):
        self.arm_left.master.set_end_torque_zero()
        self.arm_right.master.set_end_torque_zero()
        k = input('two arm move to start position?(q)')
        if k != 'q':
            self.arm_left.move_start_position()
            self.arm_right.move_start_position()

        print("move done, set end torque zero..")
        self.clear_uart()
        i = 0
        while True:
            self.record_one()
            i += 1
            self.follow()
            print('next episode？:', i)
            self.clear_uart()

    def _async_record_episode(self, info=True):
        start = time.time()
        bit_width = 20
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as exc:
            left_future = exc.submit(self.arm_left.follow, (bit_width,))
            right_future = exc.submit(self.arm_right.follow, bit_width)

            left_master_angles, left_trigger_angle, left_puppet_angles, left_grasper_angle = left_future.result()
            right_master_angles, right_trigger_angle, right_puppet_angles, right_grasper_angle = left_future.result()

        images = self.camera.read_sync()
        camera_cost = time.time() - start
        episode = {
            'left_master': left_master_angles + [left_trigger_angle],
            'left_puppet': left_puppet_angles + [left_grasper_angle],
            # 'left_trigger': left_master_trigger,
            'right_master': right_master_angles + [right_trigger_angle],
            'right_puppet': right_puppet_angles + [right_grasper_angle],
            # 'right_trigger': right_master_trigger,

            'camera': images,
            'image_size': self.camera.image_size  # H * W * 3
        }

        while (time.time() - start) < (1 / FPS):
            time.sleep(0.0001)
        duration = time.time() - start
        bit_width = 1 / duration / 2  # 时刻监控在 t>n * bit_time 情况下单条指令发送的时间

        if info:
            print(duration, "bit_width:", bit_width, "camera:", round(camera_cost, 4))
            print("left", episode["left_master"], episode["left_puppet"])
            print("right", episode["right_master"], episode["right_puppet"])
        return episode

    def _record_episode(self, info=True):
        start = time.time()
        bit_width = 20
        left_master_angles, left_trigger_angle, left_puppet_angles, left_grasper_angle = self.arm_left.follow(bit_width)
        right_master_angles, right_trigger_angle, right_puppet_angles, right_grasper_angle = self.arm_right.follow(
            bit_width)

        images = self.camera.read_sync()
        camera_cost = time.time() - start

        episode = {
            'left_master': left_master_angles + [left_trigger_angle],
            'left_puppet': left_puppet_angles + [left_grasper_angle],
            # 'left_trigger': left_master_trigger,
            'right_master': right_master_angles + [right_trigger_angle],
            'right_puppet': right_puppet_angles + [right_grasper_angle],
            # 'right_trigger': right_master_trigger,

            'camera': images,
            'image_size': self.camera.image_size  # H * W * 3
        }

        while (time.time() - start) < (1 / FPS):
            time.sleep(0.0001)
        duration = time.time() - start
        bit_width = 1 / duration / 2  # 时刻监控在 t>n * bit_time 情况下单条指令发送的时间

        if info:
            print(duration, "bit_width:", bit_width, "camera:", round(camera_cost, 4))
            print("left", episode["left_master"], episode["left_puppet"])
            print("right", episode["right_master"], episode["right_puppet"])
        return episode

    def record_one(self):
        print('start record now?')
        button.block_waiting_press()

        episodes = []

        for i in range(3):
            images = self.camera.read_sync()

        start_tm = time.time()
        while 1:
            episode = self._record_episode()
            episodes.append(episode)

            if button.is_press():
                button.reset_input_buffer()
                break
        duration = time.time() - start_tm
        f = 'output/%s/%s.pkl' % (self.folder_name, datetime.now().strftime("%m_%d_%H_%M_%S"))
        pickle.dump(episodes, open(f, 'wb'))
        print(f'save to {f}, length {len(episodes)} FPS {round(len(episodes) / duration, 2)}')

    def follow(self):
        while 1:
            self._record_episode(False)
            if button.is_press():
                break
        self.arm_left.lock()
        self.arm_right.lock()


if __name__ == '__main__':
    arm_left, arm_right = build_two_arm(TASK_CONFIG["Pick_Pen"])
    r = Recorder(arm_left, arm_right)
    button = Button(BUTTON_NAME, 9600)
    r.record()
