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

from constant import FPS, BAUDRATE, GRASPER_NAME, IMAGE_H, IMAGE_W, COM_NAME, FOLLOWERS_R, CAMERA_TOP, CAMERA_RIGHT, LEADERS_R
from dr.DrRobot import Robot, Puppet, Master
from dr.gripper import Grasper


class Recorder:

    def __init__(self):
        self.folder_name = "%s" % (datetime.now().strftime("%m_%d"))
        os.makedirs("output/%s" % self.folder_name, exist_ok=True)
        ser_port = serial.Serial(GRASPER_NAME, BAUDRATE)
        print("init grasper")
        self.dr = DrEmpower_can(com=COM_NAME, uart_baudrate=BAUDRATE)
        self.robotPuppetRight = Puppet(FOLLOWERS_R, self.dr, Grasper(ser_port, 1))
        print("init robotPuppetRight")
        self.robotMasterRight = Master(LEADERS_R, self.dr)

        self.top_cap = cv2.VideoCapture(CAMERA_TOP, cv2.CAP_DSHOW)  # top
        self.top_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.top_cap.set(3, IMAGE_W)
        self.top_cap.set(4, IMAGE_H)
        print("init top_cap")
        self.right_cap = cv2.VideoCapture(CAMERA_RIGHT, cv2.CAP_DSHOW)  # right
        self.right_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.right_cap.set(3, IMAGE_W)
        self.right_cap.set(4, IMAGE_H)
        print("init right_cap")

    def change_right_gripper(self, event):
        self.robotPuppetRight.change_gripper()

    # def open_gripper(self, event):
    #     self.robotPuppet.open_gripper()

    def recording_end(self, event):
        self.recording = False

    def record(self):

        self.robotMasterRight.set_handle_zero()
        self.robotMasterRight.set_torque([0, -0.2, 0.1, 0, 0, 0])
        # self.robotMasterRight.impedance_control(2, angle=0, speed=1, tff=0, kp=0.02, kd=0.02)
        self.robotMasterRight.impedance_control(3, angle=0, speed=1, tff=0, kp=0.02, kd=0.02)

        self.recording = False
        keyboard.on_press_key('v', self.recording_end)
        # keyboard.hook_key('v', self.close_gripper)
        keyboard.on_press_key('b', self.change_right_gripper)

        print('begin recording?')
        keyboard.wait('v')
        self.dr.clear_uart()
        i = 0
        while True:
            self.record_one()
            i += 1
            print('next episode？:', i)
            keyboard.wait('v')
            self.dr.clear_uart()

    def get_master_angles(self):
        angles = self.dr.get_angle_speed_torque_all([i for i in range(1, 13)])
        angles = [row[0] for row in angles]
        return None, None, angles[:6], angles[6:]

    def get_master_do_puppet(self):
        _, _, right_master, right_puppet = self.get_master_angles()
        print(f'right_master {right_master}')
        self.robotPuppetRight.move_to(right_master)

    def record_one(self):
        self.get_master_do_puppet()

        print('start now?')
        keyboard.wait('v')

        episode = []

        self.get_master_do_puppet()
        start = time.time()
        for i in range(5):
            # 2.获取图像
            ret, image = self.top_cap.read()
            assert ret
            retval, buffer = cv2.imencode('.jpg', image)
            ret, image = self.right_cap.read()
            assert ret
            retval, buffer2 = cv2.imencode('.jpg', image)
        while (time.time() - start) < (1 / FPS):  # t/n=10, sleep 10毫秒
            time.sleep(0.0001)
        a = time.time() - start
        bit_width = 1 / a / 2
        self.recording = True
        while self.recording:
            start = time.time()
            # 1.获取puppet角度

            # 2.获取图像
            camera_cost = time.time()
            ret, image = self.right_cap.read()
            retval, buffer = cv2.imencode('.jpg', image)
            # print(f'right:{time.time() - camera_cost}')
            camera_cost = time.time()
            ret, image = self.top_cap.read()
            retval, buffer3 = cv2.imencode('.jpg', image)
            # print(f'top:{time.time() - camera_cost}')
            # camera_cost = time.time()
            camera_cost = time.time() - camera_cost
            angles_cost = time.time()

            _, _, right_master, right_puppet = self.get_master_angles()

            episode.append({
                'right': buffer.tobytes(),
                'top': buffer3.tobytes(),
                'right_puppet': right_puppet,
                'right_master': right_master,
                'right_gripper': self.robotPuppetRight.gripper_status,
            })

            angles_cost = time.time() - angles_cost
            move_cost = time.time()

            # angle_list = self.robotMasterLeft.get_angles()
            self.robotPuppetRight.move_to2(right_master, bit_width)
            move_cost = time.time() - move_cost

            while (time.time() - start) < (1 / FPS):
                time.sleep(0.0001)
            bit_width = 1 / (time.time() - start) / 2  # 时刻监控在 t>n * bit_time 情况下单条指令发送的时间
            print((time.time() - start), right_master, self.robotPuppetRight.gripper_status, bit_width,
                  f'camera_cost:{camera_cost}, angles_cost:{angles_cost}, move_cost:{move_cost}')
        episode_name = datetime.now().strftime("%m_%d_%H_%M_%S")
        f = 'output/%s/%s.pkl' % (self.folder_name, episode_name)
        pickle.dump(episode, open(f, 'wb'))
        print(f'save to {f}, length {len(episode)}')


if __name__ == '__main__':
    r = Recorder()
    r.record()
