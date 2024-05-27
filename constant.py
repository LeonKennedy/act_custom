#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: constant.py
@time: 2024/5/13 10:20
@desc:
"""
import sys

# COM_NAME = 'COM3' # 在 windows 下控制一体化关节，相应的输入连接的COM口和波特率
# COM_NAME = '/dev/ttyAMA0' # 在树莓派（raspbian）下控制一体化关节，相应的输入连接的串口
# COM_NAME ='/dev/ttyUSB0' # 在 jetson nano（ubuntu） 下控制一体化关节，相应的输入连接的串口
# COM_NAME = "/dev/cu.usbmodem7C968E8F06521"
if sys.platform == "darwin":
    COM_NAME = "/dev/tty.usbmodem7C968E8F06521"
    GRASPER_NAME = '/dev/tty.wchusbserial14330'
    CAMERA_NAME = {"TOP": 1, "RIGHT_CAMERA": 0}
else:
    COM_NAME = "COM5"
    GRASPER_NAME = 'COM4'
    CAMERA_NAME = {"TOP": 1, "RIGHT_CAMERA": 0}

BAUDRATE = 115200  # 串口波特率，与CAN模块的串口波特率一致，（出厂默认为 115200，最高460800）

LEADERS_R = [1, 2, 3, 4, 5, 6]
# 1
FOLLOWERS_R = [7, 8, 9, 10, 11, 12]


IMAGE_W = 640
IMAGE_H = 360

FPS = 50
