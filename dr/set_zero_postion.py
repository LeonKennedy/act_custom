import time
from typing import List

from DrEmpower_can import DrEmpower_can
from constants import BAUDRATE, FOLLOWERS_L, FOLLOWERS_R, LEADERS_L, LEADERS_R, COM_LEFT, COM_RIGHT


def _set_left_servo_zero(id_list: List[int]):
    for i in id_list:
        left_dr.set_zero_position(i)
        time.sleep(2)


def _set_right_servo_zero(id_list: List[int]):
    for i in id_list:
        right_dr.set_zero_position(i)
        time.sleep(2)


def set_left_puppet():
    _set_left_servo_zero(FOLLOWERS_L)


def set_left_master():
    _set_left_servo_zero(LEADERS_L)


def set_right_puppet():
    _set_right_servo_zero(FOLLOWERS_R)


def set_right_master():
    _set_right_servo_zero(LEADERS_R)


def _get_angles(dr, id_list: List[int]):
    while 1:
        state = dr.get_angle_speed_torque_all(id_list)
        if state:
            angles = [i[0] for i in state]
            return angles[:6], angles[6:]


def get_all_angles():
    p = left_dr.get_angle_speed_torque_all(LEADERS_L + FOLLOWERS_L)
    print(p)


def get_left_angle():
    master, puppet = _get_angles(left_dr, LEADERS_L + FOLLOWERS_L)
    print("LEFT MASTER", master)
    print("LEFT PUPPET", puppet)


def get_right_angle():
    master, puppet = _get_angles(right_dr, LEADERS_R + FOLLOWERS_R)
    print("RIGHT MASTER", master)
    print("RIGHT PUPPET", puppet)


if __name__ == '__main__':
    left_dr = DrEmpower_can(com=COM_LEFT, uart_baudrate=BAUDRATE)
    right_dr = DrEmpower_can(com=COM_RIGHT, uart_baudrate=BAUDRATE)
