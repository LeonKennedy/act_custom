import abc
import time
from io import StringIO
from typing import Tuple, List
import sys

from loguru import logger

from .new_arm import build_left, build_right, Puppet, Master
from .trigger import build_trigger, Trigger
from .grasper import Grasper, build_grasper


class Arm:
    def __init__(self, m: Master, p: Puppet, trigger: Trigger, grasper: Grasper):
        self.master = m
        self.puppet = p
        self.all_id = m.id_list + p.id_list
        self.dr = p.dr
        self.tmp_buffer = StringIO()
        self.trigger = trigger
        self.grasper = grasper
        # self.dr.disable_angle_speed_torque_state()

    def get_all_angle(self) -> Tuple[List, List]:
        n = 0
        while 1:
            state = self.dr.get_angle_speed_torque_all(id_list=self.all_id)
            if state:
                angles = [i[0] for i in state]
                return angles[:6], angles[6:]
            else:
                n += 1
                time.sleep(0.002)
                logger.warning(f"read state empty times: {n}")

    def follow(self, bit_width: float = 15):
        master_angles, puppet_angles = self.get_all_angle()
        self.gravity(master_angles)
        self.puppet.move_to(master_angles, bit_width)
        self.grasper.set_angle_by_ratio(self.trigger.read())

    def gravity(self, angles):
        sys.stdout = self.tmp_buffer
        self.dr.gravity_compensation(pay_load=0, F=[0, 0, 0], angle_list=angles)
        sys.stdout = sys.__stdout__

    @abc.abstractmethod
    def move_start_position(self):
        pass


class ArmLeft(Arm):
    def __init__(self, trigger, grasper):
        m, p = build_left()
        super().__init__(m, p, trigger, grasper)

    def move_start_position(self):
        self.master.move_to1([0, 20, -30, -20, 90, 0])
        time.sleep(2)
        self.puppet.move_to1([0, 20, -30, -20, 90, 0])


class ArmRight(Arm):

    def __init__(self, trigger, grasper):
        m, p = build_right()
        super().__init__(m, p, trigger, grasper)

    def move_start_position(self):
        start = [-80, -10, 30, -22, -86, 0]
        self.master.move_to1(start)
        time.sleep(2)
        self.puppet.move_to1(start)


def build_two_arm() -> Tuple[Arm, Arm]:
    left_trigger, right_trigger = build_trigger()
    left_grasper, right_grasper = build_grasper()
    left_arm = ArmLeft(left_trigger, left_grasper)
    right_arm = ArmRight(right_trigger, right_grasper)
    return left_arm, right_arm
