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
import keyboard
import serial
import cv2

from dr import DrEmpower_can
from dr.constants import FPS, BAUDRATE, GRASPER_NAME, IMAGE_H, IMAGE_W, COM_NAME, CAMERA_TOP, CAMERA_RIGHT, LEADERS_R
from dr.DrRobot import PuppetRight, MasterRight, PuppetLeft, MasterLeft, build_arm
from button import Button
from camera import CameraGroup
from my_utils import get_angle_all


class Recorder:

    def __init__(self):
        self.folder_name = "%s" % (datetime.now().strftime("%m_%d"))
        os.makedirs("output/%s" % self.folder_name, exist_ok=True)
        self.dr = DrEmpower_can(com=COM_NAME, uart_baudrate=BAUDRATE)
        self.master_left, self.puppet_left, self.master_right, self.puppet_right = build_arm(self.dr)

        self.camera = CameraGroup()

    def record(self):
        self.master_left.begin_for_operate()
        self.master_right.begin_for_operate()

        print('begin recording?')
        button.block_waiting_press()
        self.dr.clear_uart()
        i = 0
        while True:
            self.record_one()
            i += 1
            print('next episode？:', i)
            button.block_waiting_press()
            self.dr.clear_uart()

    def get_master_do_puppet(self):
        lm, lp, _, rm, rp, _ = get_angle_all(self.dr)
        print(f'left master: {lm}, right master {rm}')
        self.puppet_left.move_to(lm)
        self.puppet_right.move_to(rm)

    def record_one(self):
        self.get_master_do_puppet()

        print('start now?')
        button.block_waiting_press()

        episodes = []

        self.get_master_do_puppet()
        start = time.time()
        for i in range(3):
            images = self.camera.read_sync()

        while (time.time() - start) < (1 / FPS):  # t/n=10, sleep 10毫秒
            time.sleep(0.0001)
        a = time.time() - start
        bit_width = 1 / a / 2

        while 1:
            start = time.time()
            # 2.获取图像
            camera_cost = time.time()
            images = self.camera.read_sync()
            camera_cost = time.time() - camera_cost

            lm, lp, left_puppet_gripper, rm, rp, right_puppet_gripper = get_angle_all(self.dr)
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
            episodes.append(episode)

            self.puppet_left.set_gripper_ratio(left_master_trigger)
            self.puppet_right.set_gripper_ratio(right_master_trigger)
            self.puppet_right.move_to2(rm, bit_width)
            self.puppet_left.move_to2(lm, bit_width)

            while (time.time() - start) < (1 / FPS):
                time.sleep(0.0001)
            bit_width = 1 / (time.time() - start) / 2  # 时刻监控在 t>n * bit_time 情况下单条指令发送的时间
            print((time.time() - start), "bit_width:", bit_width, "camera:", round(camera_cost, 4))
            print("left", episode["left_master"], episode["left_puppet"])
            print("right", episode["right_master"], episode["right_puppet"])

            if button.is_press():
                button.reset_input_buffer()
                break
        f = 'output/%s/%s.pkl' % (self.folder_name, datetime.now().strftime("%m_%d_%H_%M_%S"))
        pickle.dump(episodes, open(f, 'wb'))
        print(f'save to {f}, length {len(episodes)}')


if __name__ == '__main__':
    r = Recorder()
    button = Button("COM6", 9600)
    r.record()
