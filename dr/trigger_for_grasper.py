import time

from trigger import build_trigger
from grasper import build_grasper
from constants import TRIGGER_NAME, GRASPER_NAME

from collections import deque
import numpy as np


def run():
    r_torque_q = deque(maxlen=30)
    l.set_pwm(650)
    r.set_pwm(650)
    while 1:
        left_angle_ratio_of_trigger = l.read()
        right_angle_ratio_of_trigger = r.read()
        print(left_angle_ratio_of_trigger, l.position(), right_angle_ratio_of_trigger, r.position())
        lf.set_angle_by_ratio(left_angle_ratio_of_trigger)
        rf.set_angle_by_ratio(right_angle_ratio_of_trigger)

        r_torque_q.append(rf.read_torque())
        # print("TORQUE:", lf.read_torque(), r_torque_q[-1], np.mean(r_torque_q))
        # set pwm
        # l.set_pwm(lf.read_torque() / 2)
        # r.set_pwm(np.mean(r_torque_q) / 2)
        time.sleep(1 / 40)


if __name__ == '__main__':
    lf, rf = build_grasper(GRASPER_NAME)
    l, r = build_trigger(TRIGGER_NAME)
    run()
