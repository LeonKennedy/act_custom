import abc
import time
from io import StringIO
from typing import Tuple, List, Dict
import sys

from loguru import logger

from .new_arm import build_master_and_puppet, Puppet, Master
from .trigger import build_trigger, Trigger
from .grasper import Grasper, build_grasper
from .constants import COM_LEFT, COM_RIGHT, LEADERS_L, LEADERS_R, FOLLOWERS_L, FOLLOWERS_R, TRIGGER_NAME, GRASPER_NAME


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
                # logger.warning(f"{self.__class__} read state empty times: {n}")

    def follow(self, bit_width: float = 15):
        master_angles, puppet_angles = self.get_all_angle()
        grasper_angle = self.grasper.read_angle()
        trigger_angle = self.grasper.ratio_to_angle(self.trigger.read())
        self.gravity(master_angles)
        self.puppet.move_to(master_angles, bit_width)
        self.grasper.set_angle(trigger_angle)
        return master_angles, trigger_angle, puppet_angles, grasper_angle

    def gravity(self, angles):
        sys.stdout = self.tmp_buffer
        self.dr.gravity_compensation(pay_load=0, F=[0, 0, 0], angle_list=angles)
        sys.stdout = sys.__stdout__

    def set_angle(self, sid: int, angle: float):
        self.dr.set_angle(sid, angle, 10, 10, 1)

    def step(self, sid: int, angle: float):
        self.dr.step_angle(sid, angle, 10, 10, 1)

    def set_zero_position(self, sid):
        self.dr.set_zero_position(sid)

    def lock(self):
        master_angles, puppet_angles = self.get_all_angle()
        # last not lock
        self.dr.set_angles(self.master.id_list[:-1], master_angles[:-1], 10, 10, 1)
        print(self.__class__, "lock at", master_angles)

    @abc.abstractmethod
    def move_start_position(self):
        pass

    def clear_uart(self):
        self.dr.uart.flushInput()


class ArmLeft(Arm):
    def __init__(self, trigger, grasper):
        print("init left arm")
        m, p = build_master_and_puppet(COM_LEFT, master_ids=LEADERS_L, puppet_ids=FOLLOWERS_L)
        super().__init__(m, p, trigger, grasper)

    def move_start_position(self):
        start = [-35, 15, -78, -20, 90, 0-14]
        self.master.move_to1(start)
        time.sleep(2)
        self.puppet.move_to1(start)


class ArmRight(Arm):

    def __init__(self, trigger, grasper):
        print("init right arm")
        m, p = build_master_and_puppet(COM_RIGHT, master_ids=LEADERS_R, puppet_ids=FOLLOWERS_R)
        super().__init__(m, p, trigger, grasper)

    def move_start_position(self):
        start = [32, 16, 90, -3, -86, -6]
        self.master.move_to1(start)
        time.sleep(2)
        self.puppet.move_to1(start)


def build_two_arm() -> Tuple[Arm, Arm]:
    left_trigger, right_trigger = build_trigger(TRIGGER_NAME)
    left_grasper, right_grasper = build_grasper(GRASPER_NAME)
    left_arm = ArmLeft(left_trigger, left_grasper)
    right_arm = ArmRight(right_trigger, right_grasper)
    return left_arm, right_arm
