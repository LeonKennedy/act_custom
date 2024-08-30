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
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import keyboard

from dr import build_two_arm, Arm, fps_wait
from dr.constants import FPS
from camera import CameraGroup
from prompt import cube_assemble

BUTTON_KEY = '5'


class Recorder:

    def __init__(self, arm_left: Arm, arm_right: Arm):
        self.save_path = "output/%s/%s" % (task_name, datetime.now().strftime("%m_%d"))
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

        self.arm_left = arm_left
        self.arm_right = arm_right
        self.camera = CameraGroup()
        self.bit_width = 1 / FPS / 2
        self.record_frequency = 2
        print("Moving FPS", FPS, "Recording FPS", FPS / self.record_frequency)

    def clear_uart(self):
        self.arm_left.clear_uart()
        self.arm_right.clear_uart()

    def record(self):
        k = input('[DO FIRST]\n1. two arm move to start position?\n2. master move to puppet?(q)')
        if k == '1':
            self.arm_left.move_start_position()
            self.arm_right.move_start_position()
        elif k == '2':
            self.master_to_puppet()
        else:
            pass

        self.arm_left.master.set_end_torque_zero()
        self.arm_right.master.set_end_torque_zero()
        print("move done, set end torque zero..")
        task = cube_assemble()
        print("TASK: ", task)
        self.clear_uart()
        i = 0
        global RUNNING_FLAG
        keyboard.on_press_key(BUTTON_KEY, _change_running_flag)
        while True:
            self.record_one(task)
            i += 1
            RUNNING_FLAG = True
            self.follow()
            print('next episode？:', i)
            self.clear_uart()

    # def _async_record_episode(self, info=True):
    #     start = time.time()
    #     bit_width = 1 / FPS / 2
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as exc:
    #         left_future = exc.submit(self.arm_left.follow, (bit_width,))
    #         right_future = exc.submit(self.arm_right.follow, bit_width)
    #
    #         left_master_angles, left_trigger_angle, left_puppet_angles, left_grasper_angle = left_future.result()
    #         right_master_angles, right_trigger_angle, right_puppet_angles, right_grasper_angle = left_future.result()
    #
    #     images = self.camera.read_sync()
    #     camera_cost = time.time() - start
    #     episode = {
    #         'left_master': left_master_angles + [left_trigger_angle],
    #         'left_puppet': left_puppet_angles + [left_grasper_angle],
    #         # 'left_trigger': left_master_trigger,
    #         'right_master': right_master_angles + [right_trigger_angle],
    #         'right_puppet': right_puppet_angles + [right_grasper_angle],
    #         # 'right_trigger': right_master_trigger,
    #
    #         'camera': images,
    #         'image_size': self.camera.image_size  # H * W * 3
    #     }
    #
    #     while (time.time() - start) < (1 / FPS):
    #         time.sleep(0.0001)
    #     duration = time.time() - start
    #     bit_width = 1 / duration / 2  # 时刻监控在 t>n * bit_time 情况下单条指令发送的时间
    #
    #     if info:
    #         print(duration, "bit_width:", bit_width, "camera:", round(camera_cost, 4))
    #         print("left", episode["left_master"], episode["left_puppet"])
    #         print("right", episode["right_master"], episode["right_puppet"])
    #     return episode

    def _record_episode(self, info=True):
        start = time.time()
        left_master_angles, left_trigger_angle, left_puppet_angles, left_grasper_angle = self.arm_left.follow(
            self.bit_width)
        right_master_angles, right_trigger_angle, right_puppet_angles, right_grasper_angle = self.arm_right.follow(
            self.bit_width)
        tm1 = time.time()
        images = self.camera.read_sync()
        camera_cost = time.time() - tm1

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

        fps_wait(FPS, start)
        duration = time.time() - start
        self.bit_width = 1 / duration / 2  # 时刻监控在 t>n * bit_time 情况下单条指令发送的时间

        if info:
            print(duration, "bit_width:", self.bit_width, "camera:", round(camera_cost, 4))
            print("left", episode["left_master"], episode["left_puppet"])
            print("right", episode["right_master"], episode["right_puppet"])
        return episode

    def record_one(self, task: list):
        print('start record now?')
        keyboard.wait(BUTTON_KEY)
        episodes = []

        for i in range(3):
            images = self.camera.read_sync()

        start_tm = time.time()
        i = 0
        while RUNNING_FLAG:
            episode = self._record_episode()
            if i % self.record_frequency == 0:
                episodes.append(episode)
            i += 1

        duration = time.time() - start_tm
        f = f'{self.save_path}/{datetime.now().strftime("%m_%d_%H_%M_%S")}.pkl'
        pickle.dump({"data": episodes, "task": task}, open(f, 'wb'))
        print(f'save to {f}, length {len(episodes)} FPS {round(len(episodes) / duration, 2)}')

    def follow(self):
        while RUNNING_FLAG:
            self._record_episode(False)

        self.arm_left.lock()
        self.arm_right.lock()

    def master_to_puppet(self):
        lm, lp = self.arm_left.get_all_angle()
        self.arm_left.master.move_to1(lp)
        rm, rp = self.arm_right.get_all_angle()
        self.arm_right.master.move_to1(rp)


RUNNING_FLAG = False


def _change_running_flag(event):
    global RUNNING_FLAG
    RUNNING_FLAG = not RUNNING_FLAG
    print(f"change running flag to {RUNNING_FLAG}")


if __name__ == '__main__':
    task_name = sys.argv[1]
    arm_left, arm_right = build_two_arm()
    r = Recorder(arm_left, arm_right)
    r.record()
