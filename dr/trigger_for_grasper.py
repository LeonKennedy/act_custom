import time

from trigger import build_trigger
from grasper import build_grasper
from constants import TRIGGER_NAME


def run():
    while 1:
        left_angle_ratio_of_trigger = l.read()
        right_angle_ratio_of_trigger = r.read()
        print(left_angle_ratio_of_trigger, l.position(), right_angle_ratio_of_trigger, r.position())
        lf.set_angle_by_ratio(left_angle_ratio_of_trigger)
        rf.set_angle_by_ratio(right_angle_ratio_of_trigger)
        time.sleep(1 / 40)


if __name__ == '__main__':
    lf, rf = build_grasper({})
    l, r = build_trigger(TRIGGER_NAME)
    run()
