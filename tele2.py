import time

from button import Button
from dr import ArmLeft, build_two_arm
from dr.constants import BUTTON_NAME


def left_follow():
    time.sleep(2)
    arm_left.master.set_end_torque_zero()
    print("[[   Start   ]]")
    bit_width = 30
    while 1:
        sm = time.time()
        arm_left.follow(bit_width)
        time.sleep(0.002)
        bit_width = 1 / (time.time() - sm) / 2


def right_follow():
    time.sleep(2)
    arm_right.master.set_end_torque_zero()
    print("[[   Start   ]]")
    bit_width = 20
    while 1:
        sm = time.time()
        arm_right.follow(20)
        time.sleep(0.002)
        # bit_width = 1 / (time.time() - sm) / 2


def start_position():
    arm_left.move_start_position()
    arm_right.move_start_position()


def follow():
    button = Button(BUTTON_NAME, 9600)
    arm_left.master.set_end_torque_zero()
    arm_right.master.set_end_torque_zero()
    print("start?")
    button.block_waiting_press()
    fps = 40
    bit_width = fps / 2
    while 1:
        arm_left.follow(bit_width)
        arm_right.follow(bit_width)
        time.sleep(1 / fps)

        if button.is_press():
            break
    arm_left.master.lock()
    arm_right.master.lock()


if __name__ == '__main__':
    arm_left, arm_right = build_two_arm()
