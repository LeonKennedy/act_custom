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
from dr.DrRobot import PuppetRight, MasterRight, MasterLeft, PuppetLeft
from dr.constants import COM_NAME, BAUDRATE, TRIGGER_NAME
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
    print("start init follow")
    puppet_right.move_zero()
    puppet_left.move_zero()
    master_right.move_zero()
    master_left.move_zero()
    print("move to zero")

    master_right.begin_for_operate()
    master_left.begin_for_operate()
    print("start?")
    f = button.block_waiting_press()
    while 1:
        lm, lp, rm, rp, = get_angle_all(dr)
        logger.info(f"{lm=} {lp=} {rm} {rp}")
        puppet_right.move_to2(rm, bit_width)
        puppet_right.set_gripper_ratio(master_right.read())
        puppet_left.move_to2(lm, bit_width)
        puppet_left.set_gripper_ratio(master_left.read())

        if button.is_press():
            break
        time.sleep(1 / 20)
    print("end follow")


def right_follow():
    print("start right follow")
    puppet_right.move_zero()
    master_right.move_zero()
    print("move to zero")

    master_right.begin_for_operate()
    s = input("start?(q)")
    if s == 'q':
        return

    while 1:
        lm, lp, rm, rp, = get_angle_all(dr)
        logger.info(f"{lm=} {lp=} {rm} {rp}")
        puppet_right.move_to2(rm, bit_width)
        puppet_right.set_gripper_ratio(master_right.read())
        time.sleep(1 / 20)


def left_follow():
    print("start left follow")
    puppet_left.move_zero()
    master_left.move_zero()
    print("move to zero")

    master_left.begin_for_operate()
    s = input("start?(q)")
    if s == 'q':
        return

    while 1:
        master, puppet, _, _ = get_angle_all(dr)
        logger.info(f"{master=} {puppet=}")
        puppet_left.move_to2(master, bit_width)
        puppet_left.set_gripper_ratio(master_left.read())
        time.sleep(1 / 20)


def test_trigger():
    while 1:
        l = master_left.trigger.read()
        r = master_right.trigger.read()
        puppet_left.set_gripper_ratio(l)
        puppet_right.set_gripper_ratio(r)
        print("left torque:", dr.get_torque(25), "right_torque:", dr.get_torque(26))
        time.sleep(1 / 50)


def info(sid: int):
    print("angle", round(dr.get_angle(sid), 4), "torque", round(dr.get_torque(sid), 4),
          "speed", round(dr.get_speed(sid), 4))


def stop():
    for i in (20, 8):
        dr.set_torque(i, dr.get_torque(i) * 0.95)


if __name__ == '__main__':
    dr = DrEmpower_can(com=COM_NAME, uart_baudrate=BAUDRATE)
    # dr.set_torque_limit(0, 6)
    puppet_right = PuppetRight(dr)
    puppet_left = PuppetLeft(dr)
    #
    left_trigger, right_trigger = build_trigger(TRIGGER_NAME)
    master_left = MasterLeft(dr, left_trigger)
    master_right = MasterRight(dr, right_trigger)

    # ser_port = serial.Serial(GRASPER_NAME, BAUDRATE)
    # master_right = MasterRight(dr)
    # # puppet_right = PuppetRight(dr, None)
    # puppet_right = PuppetRight(dr, Grasper(ser_port, GRIPPER_RIGHT_ID, max_v=1550))
    # running_flag = True
    bit_width = get_bit_width()
    button = Button("COM6", 9600)

    # dr.impedance_control(2, angle=10, speed=1, tff=0, kp=0.05, kd=0.05)
    # dr.impedance_control(3, angle=-30, speed=1, tff=0, kp=0.05, kd=0.05)
    # leader_arm.release_torque_for_operate()
    dr.disable_angle_speed_torque_state()
    # follow_arm.init_positions()
    # dr.motion_aid(id_num=id_num, angle=0, speed=5, angle_err=1, speed_err=1, torque=8)
