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
from loguru import logger

from button import Button
from dr import DrEmpower_can
from dr.DrRobot import PuppetRight, MasterRight, MasterLeft, PuppetLeft, build_arm
from dr.constants import COM_NAME, BAUDRATE, TRIGGER_NAME, BUTTOM_NAME
from dr.dx_trigger import build_trigger
from my_utils import get_angle_all
from dr.sj_gripper import Grasper


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
    puppet_right.move_to([0, 0, 0, 0, 0, 0], wait=True)

    master_right.begin_for_operate()

    keyboard.on_press_key("b", puppet_right.change_gripper)
    keyboard.on_press_key("s", change_running_callback)
    # dr.disable_angle_speed_torque_state(id_num=0)
    inp = input("start ?(q)")
    if inp != "q":
        follow()
    # self.robotMaster.impedance_control(8, angle=60, speed=1, tff=0, kp=0.05, kd=0.05)


def change_running_callback(e):
    global running_flag
    running_flag = not running_flag


def follow():
    button = Button(BUTTOM_NAME, 9600)
    print("start move?")
    f = button.block_waiting_press()
    master_left.gravity()
    master_right.gravity()
    print("start follow?")
    f = button.block_waiting_press()
    while 1:
        stm = time.time()
        lp, rp, l_g, r_g = get_angle_all(dr)
        logger.info(f"{lp=} {rp=} {l_g} {r_g}")

        master_right.gravity()
        master_left.gravity()
        rm = master_right.read_angles()
        lm = master_left.read_angles()

        bit_width = 1 / (time.time() - stm) / 4  # 计算轨迹跟踪模式下指令发送带宽
        puppet_right.move_to2(rm, bit_width)
        puppet_right.set_gripper_ratio(master_right.trigger.read())

        puppet_left.move_to2(lm, bit_width)
        puppet_left.set_gripper_ratio(master_left.trigger.read())

        if button.is_press():
            break
    master_left.out_gravity()
    master_right.out_gravity()
    print("end follow")


def right_follow():
    master_right.gravity()
    s = input("start right?(q)")
    if s == 'q':
        return

    while 1:
        stm = time.time()
        lp, rp, lg, rg = get_angle_all(dr)
        logger.info(f"{lp=} {lg=} {rp=} {rg=}")
        master_right.gravity()
        rm = master_right.read_angles()

        inter = (time.time() - stm)
        bit_width = 1 / inter / 2  # 计算轨迹跟踪模式下指令发送带宽
        puppet_right.move_to2(rm, bit_width)
        puppet_right.set_gripper_ratio(master_right.trigger.read())
        # time.sleep(1 / 20)


def left_follow():
    master_left.gravity()
    s = input("start?(q)")
    if s == 'q':
        return

    while 1:
        stm = time.time()
        lp, rp, lg, rg = get_angle_all(dr)
        logger.info(f"{lp=} {lg=} {rp=} {rg=}")

        master_left.gravity()
        lm = master_left.read_angles()
        bit_width = 1 / (time.time() - stm) / 2
        puppet_left.move_to2(lm, bit_width)
        puppet_left.set_gripper_ratio(master_left.trigger.read())


def test_trigger():
    while 1:
        l = master_left.trigger.read()
        r = master_right.trigger.read()
        puppet_left.set_gripper_ratio(l)
        puppet_right.set_gripper_ratio(r)
        print("left torque:", dr.get_torque(puppet_left.gripper_id), "right torque:",
              dr.get_torque(puppet_right.gripper_id))
        time.sleep(1 / 50)


def info(sid: int):
    print("angle", round(dr.get_angle(sid), 4), "torque", round(dr.get_torque(sid), 4),
          "speed", round(dr.get_speed(sid), 4))


def stop():
    for i in (20, 8):
        dr.set_torque(i, dr.get_torque(i) * 0.95)


if __name__ == '__main__':
    dr = DrEmpower_can(com="COM3", uart_baudrate=BAUDRATE)
    dr.disable_angle_speed_torque_state()
    # dr.set_torque_limit(0, 6)
    # #
    master_left, puppet_left, master_right, puppet_right = build_arm(dr)
    # # ser_port = serial.Serial(GRASPER_NAME, BAUDRATE)
    # # master_right = MasterRight(dr)
    bit_width = get_bit_width()
    #

    # dr.impedance_control(2, angle=10, speed=1, tff=0, kp=0.05, kd=0.05)
    # dr.impedance_control(3, angle=-30, speed=1, tff=0, kp=0.05, kd=0.05)
    # leader_arm.release_torque_for_operate()
    # follow_arm.init_positions()
    # dr.motion_aid(id_num=id_num, angle=0, speed=5, angle_err=1, speed_err=1, torque=8)
