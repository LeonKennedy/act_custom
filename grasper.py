#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: graper.py
@time: 2024/5/14 11:18
@desc:
"""
import serial
from constant import GRASPER_NAME, BAUDRATE


class Grasper:

    def __init__(self, name: str = GRASPER_NAME):
        self._min_v = 1350
        self._max_v = 1850
        self._s = serial.Serial(name, BAUDRATE)

    def _set_pwm(self, v: str):
        # 000P1800T1000!
        # data = b'#001P' + v.encode() + b'T1000!'
        data = f'#001P{v}T1000!'.encode('utf-8')
        print("send data:", data)
        self._s.write(data)

    def _set(self, ratio: float):
        v = self._min_v + int((self._max_v - self._min_v) * ratio)
        self._set_pwm(str(v))

    def clamp(self):
        self._set(1)

    def loose(self):
        self._set(0)


if __name__ == '__main__':
    name = GRASPER_NAME
    g = Grasper(name)
