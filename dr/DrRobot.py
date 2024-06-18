from typing import List, Optional
from loguru import logger

from .DrEmpower_can import DrEmpower_can
from .dx_trigger import Trigger, build_trigger
from .constants import LEADERS_R, FOLLOWERS_R, LEADERS_L, FOLLOWERS_L, GRIPPER_RANGE_MAX, TRIGGER_NAME


class Robot:
    def __init__(self, id_list: List, dr: DrEmpower_can):
        self.id_list = id_list
        self.dr = dr

    def _check_is_have(self, sid: int) -> bool:
        if sid in self.id_list:
            return True
        print(f"[Warning] id {sid} not in {self.__class__.__name__} arm!!")
        return False

    def set_torque(self, torque_list):
        self.dr.disable_angle_speed_torque_state(id_num=0)  # 先全部取消状态反馈

        self.dr.set_torques(id_list=self.id_list, torque_list=torque_list, param=3, mode=0)

    def impedance_control(self, id_num=0, angle=0, speed=0, tff=0, kp=0, kd=0):
        self.dr.impedance_control(id_num, angle=angle, speed=speed, tff=tff, kp=kp, kd=kd)

    def set_zero(self, sid: Optional[int] = None):
        if self._check_is_have(sid):
            self.dr.set_zero_position(sid)

    def step(self, sid: int, angle: int):
        if self._check_is_have(sid):
            self.dr.step_angle(sid, angle, speed=10, param=10, mode=1)

    def set_angle(self, sid: int, angle: int):
        if self._check_is_have(sid):
            self.dr.set_angle(sid, angle, speed=10, param=10, mode=1)

    def get_angle(self, sid: int):
        return self.dr.get_angle(sid)

    def move_zero(self):
        self.dr.set_angles(id_list=self.id_list, angle_list=[0, 0, 0, 0, 0, 0], speed=10, param=10,
                           mode=1)

    def move_to(self, angle_list, wait=False):
        print('move_to:', angle_list)
        self.dr.set_angles(id_list=self.id_list, angle_list=angle_list, speed=10, param=10,
                           mode=1)  # 先控制关节已梯形估计模型平缓运动到曲线起点
        if wait:
            self.dr.positions_done(self.id_list)

    def move_to2(self, angle_list, bit_width):
        self.dr.set_angles(id_list=self.id_list, angle_list=angle_list, speed=20, param=bit_width, mode=0)

    def reboot(self):
        for i in self.id_list:
            self.dr.reboot(i)
        logger.debug(f"{self.__class__.__name__} {self.id_list} reboot")

    def set_torque_zero(self):
        self.set_torque([0, 0, 0, 0, 0, 0])

    def set_zeros(self):
        for i in self.id_list:
            self.dr.set_zero_position(i)
        logger.debug(f"{self.__class__.__name__} set {self.id_list} zero position!")

    # ---         gripper   --

    @property
    def gripper_status(self):
        return self.gripper.status


class Master(Robot):

    def __init__(self, id_list: List, dr: DrEmpower_can, trigger: Trigger):
        super().__init__(id_list, dr)
        self.trigger = trigger

    def begin_for_operate(self):
        self.set_torque_zero()
        # self.impedance_control(2, angle=0, speed=1, tff=0, kp=0.001, kd=0.02)
        # self.impedance_control(3, angle=0, speed=1, tff=0, kp=0.02, kd=0.02)

    def change_impedance_control(self, i: int, angle: float):
        self.impedance_control(i, angle=angle)


class Puppet(Robot):

    def __init__(self, id_list: List, dr: DrEmpower_can):
        super().__init__(id_list[:-1], dr)
        self.gripper_id = id_list[-1]
        self.gripper_current_angle = None
        self.angle_range = (0, 0)
        self.init_gripper()

    def _check_is_have(self, sid: int) -> bool:
        if sid == self.gripper_id:
            return True
        return super()._check_is_have(sid)

    def init_gripper(self):
        self.dr.set_angle(self.gripper_id, 150, speed=10, param=10, mode=1)
        self.gripper_current_angle = 150
        self.dr.position_done(self.gripper_id)

        self.angle_range = (0, GRIPPER_RANGE_MAX)
        self.dr.set_angle_range(self.gripper_id, angle_min=self.angle_range[0], angle_max=self.angle_range[1])
        logger.debug(f"{self.__class__.__name__} set angle range {self.angle_range}")

        self.dr.set_torque_limit(self.gripper_id, 0.39)

    def ratio_to_angle(self, ratio: float):
        return self.angle_range[1] - (self.angle_range[1] - self.angle_range[0]) * ratio

    def set_gripper_ratio(self, ratio: float, max_speed: int = 30, param: int = 10):
        angle = self.ratio_to_angle(ratio)
        self.set_gripper(angle)

    def set_gripper(self, angle: float):
        if self.gripper_current_angle + 3 > angle > self.gripper_current_angle - 3:
            return
        print(self.gripper_id, angle)
        logger.debug(f"set {self.__class__.__name__} gripper {angle} speed 10 param 10 mode 0")
        self.dr.set_angle(self.gripper_id, angle, 10, 10, 0)
        self.gripper_current_angle = angle

    # def open_gripper(self, event=None):
    #     self.gripper.loose()
    #
    # def close_gripper(self, event=None):
    #     self.gripper.clamp()
    #
    # def change_gripper(self, event=None):
    #     self.gripper.change()


class MasterLeft(Master):
    def __init__(self, dr, trigger):
        super().__init__(LEADERS_L, dr, trigger)

    def begin_for_operate(self):
        super().begin_for_operate()
        self.impedance_control(self.id_list[1], angle=0, speed=1, tff=0, kp=0.02, kd=0.002)
        # self.impedance_control(self.id_list[3], angle=0, speed=1, tff=0, kp=0.005, kd=0.02)
        # self.impedance_control(self.id_list[2], angle=50, speed=1, tff=0, kp=0.005, kd=0.02)


class MasterRight(Master):

    def __init__(self, dr, trigger):
        super().__init__(LEADERS_R, dr, trigger)

    def begin_for_operate(self):
        super().begin_for_operate()
        self.impedance_control(self.id_list[1], angle=0, speed=1, tff=0, kp=0.02, kd=0.002)


class PuppetRight(Puppet):

    def __init__(self, dr):
        super().__init__(FOLLOWERS_R, dr)


class PuppetLeft(Puppet):

    def __init__(self, dr):
        super().__init__(FOLLOWERS_L, dr)


def build_arm(dr):
    puppet_right = PuppetRight(dr)
    puppet_left = PuppetLeft(dr)

    left_trigger, right_trigger = build_trigger(TRIGGER_NAME)
    master_left = MasterLeft(dr, left_trigger)
    master_right = MasterRight(dr, right_trigger)
    return master_left, puppet_left, master_right, puppet_right
