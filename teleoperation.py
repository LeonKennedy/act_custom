#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: teleoperation.py
@time: 2024/5/15 17:28
@desc:
"""
import time

import numpy as np
import serial
import keyboard

from dr import DrEmpower_can
from dr.DrRobot import PuppetRight, MasterRight
from dr.constants import COM_NAME, BAUDRATE, GRASPER_NAME, GRIPPER_RIGHT_ID
from dr.gripper import Grasper


def get_bit_width():
    t = 10  # 曲线运行大概周期
    i = 0  # 计数初始值
    n = 200  # 曲线切割的点数
    start = time.time()
    while (time.time() - start) < (t / n):
        time.sleep(0.0001)
    inter = (time.time() - start)
    bit_width = 1 / inter / 2  # 计算轨迹跟踪模式下指令发送带宽
    return bit_width


def run():
    # dr.disable_angle_speed_torque_state(id_num=0)  # 先取消试试状态反馈，以免影响后面的普通参数回读指令
    # leader_arm.set_torque([0, 0.2, 0.1, 0, 0, 0])
    puppet.move_to([0, 0, 0, 0, 0, 0], wait=True)

    master.begin_for_operate()

    keyboard.on_press_key("b", puppet.change_gripper)
    keyboard.on_press_key("s", change_running_callback)
    # dr.disable_angle_speed_torque_state(id_num=0)
    inp = input("start ?(q)")
    if inp != "q":
        follow()
    # self.robotMaster.impedance_control(8, angle=60, speed=1, tff=0, kp=0.05, kd=0.05)


def get_angle_all():
    angles = dr.get_angle_speed_torque_all([i for i in range(1, 13)])
    angles = [row[0] for row in angles]
    return None, None, angles[:6], angles[6:]


def change_running_callback(e):
    global running_flag
    running_flag = not running_flag


def follow():
    print("start follow")
    while running_flag:
        _, _, master_right_angles, puppet_right_angles = get_angle_all()
        print("Puppet", puppet_right_angles, "move to", master_right_angles)
        puppet.move_to2(master_right_angles, bit_width)
        time.sleep(1 / 20)
    print("end follow")


if __name__ == '__main__':
    dr = DrEmpower_can(com=COM_NAME, uart_baudrate=BAUDRATE)
    ser_port = serial.Serial(GRASPER_NAME, BAUDRATE)
    master = MasterRight(dr)
    puppet = PuppetRight(dr, Grasper(ser_port, GRIPPER_RIGHT_ID))
    running_flag = True
    bit_width = get_bit_width()

    #
    # dr.impedance_control(2, angle=10, speed=1, tff=0, kp=0.05, kd=0.05)
    # dr.impedance_control(3, angle=-30, speed=1, tff=0, kp=0.05, kd=0.05)
    # leader_arm.release_torque_for_operate()
    # dr.disable_angle_speed_torque_state()
    # follow_arm.init_positions()
