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
import serial
import cv2

from dr import DrEmpower_can
from dr.constants import FPS, BAUDRATE, GRASPER_NAME, IMAGE_H, IMAGE_W, COM_NAME, CAMERA_TOP, CAMERA_RIGHT, BUTTOM_NAME
from dr.DrRobot import PuppetRight, MasterRight, PuppetLeft, MasterLeft, build_arm
from button import Button
from camera import CameraGroup
from my_utils import get_angle_all


class Recorder:

    def __init__(self):
        self.folder_name = "%s" % (datetime.now().strftime("%m_%d"))
        os.makedirs("output/%s" % self.folder_name, exist_ok=True)
        self.dr = DrEmpower_can(com=COM_NAME, uart_baudrate=BAUDRATE)
        self.dr.disable_angle_speed_torque_state()
        self.master_left, self.puppet_left, self.master_right, self.puppet_right = build_arm(self.dr)

        self.camera = CameraGroup()

    def record(self):
        print('begin recording?')
        self.master_left.gravity()
        self.master_right.gravity()
        button.block_waiting_press()
        self.dr.clear_uart()
        self.follow()
        print("is ready?")
        button.block_waiting_press()
        i = 0
        while True:
            self.record_one()
            i += 1
            self.follow()
            print('next episode？:', i)
            self.dr.clear_uart()

    def start_init(self):
        for i in range(10):
            self.master_left.gravity()
            self.master_right.gravity()
            time.sleep(0.2)

    def _record_episode(self, info=True):
        start = time.time()
        self.master_right.gravity()
        self.master_left.gravity()

        images = self.camera.read_sync()
        camera_cost = time.time() - start

        lp, rp, left_puppet_gripper, right_puppet_gripper = get_angle_all(self.dr)
        lm = self.master_left.read_angles()
        rm = self.master_right.read_angles()
        left_master_trigger = self.master_left.trigger.read()
        right_master_trigger = self.master_right.trigger.read()
        episode = {
            'left_master': lm + [self.puppet_left.ratio_to_angle(left_master_trigger)],
            'left_puppet': lp + [left_puppet_gripper],
            'left_trigger': left_master_trigger,

            'right_master': rm + [self.puppet_right.ratio_to_angle(right_master_trigger)],
            'right_puppet': rp + [right_puppet_gripper],
            'right_trigger': right_master_trigger,

            'camera': images
        }

        while (time.time() - start) < (1 / FPS):
            time.sleep(0.0001)
        duration = time.time() - start
        bit_width = 1 / duration / 2  # 时刻监控在 t>n * bit_time 情况下单条指令发送的时间

        self.puppet_left.set_gripper_ratio(left_master_trigger)
        self.puppet_right.set_gripper_ratio(right_master_trigger)
        self.puppet_right.move_to2(rm, bit_width)
        self.puppet_left.move_to2(lm, bit_width)

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

        while 1:
            episode = self._record_episode()
            episodes.append(episode)

            if button.is_press():
                button.reset_input_buffer()
                break
        f = 'output/%s/%s.pkl' % (self.folder_name, datetime.now().strftime("%m_%d_%H_%M_%S"))
        pickle.dump(episodes, open(f, 'wb'))
        print(f'save to {f}, length {len(episodes)}')

    def follow(self):
        while 1:
            self._record_episode(False)
            if button.is_press():
                break
        self.master_left.out_gravity()
        self.master_right.out_gravity()


if __name__ == '__main__':
    r = Recorder()
    button = Button(BUTTOM_NAME, 9600)
    r.record()
