from typing import List, Optional
from loguru import logger

from .DrEmpower_can import DrEmpower_can
from .gripper import Grasper
from .trigger import Trigger
from .constants import LEADERS_R, FOLLOWERS_R, LEADERS_L, FOLLOWERS_L, FOLLOWERS_LEFT_ANGLE_MAX, \
    FOLLOWERS_LEFT_ANGLE_MIN


class Robot:
    def __init__(self, id_list: List, dr: DrEmpower_can, gripper: Grasper = None):
        self.id_list = id_list
        self.dr = dr
        if gripper:
            self.gripper = gripper

        self.recording = False

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
                           mode=1)  # 先控制关节已梯形估计模型平缓运动到曲线起点

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

    # ---         gripper   --

    @property
    def gripper_status(self):
        return self.gripper.status


class Master(Robot):

    def set_handle_zero(self):
        print("ser", self.id_list[-1], "set to zero position")
        self.dr.set_zero_position(self.id_list[-1])

    def begin_for_operate(self):
        pass


class Puppet(Robot):

    def open_gripper(self, event=None):
        self.gripper.loose()

    def close_gripper(self, event=None):
        self.gripper.clamp()

    def change_gripper(self, event=None):
        self.gripper.change()

    def set_gripper(self, gripper):
        if gripper == 1:
            self.open_gripper()
        elif gripper == 2:
            self.close_gripper()


class MasterLeft(Master):
    def __init__(self, dr):
        super().__init__(LEADERS_L, dr)
        trigger = Trigger()
        logger.debug(f"left trigger init, get zero is: {trigger.zero}")
        self.trigger = trigger

    def read(self) -> float:
        return self.trigger.read()

    def begin_for_operate(self):
        super().begin_for_operate()
        self.set_torque([0, -0.2, -0.1, 0, 0, 0])
        self.set_torque([0, 0, 0, 0, 0, 0])
        self.impedance_control(self.id_list[1], angle=19, speed=1, tff=0, kp=0.005, kd=0.02)
        self.impedance_control(self.id_list[2], angle=50, speed=1, tff=0, kp=0.005, kd=0.02)
        # self.impedance_control(3, angle=0, speed=1, tff=0, kp=0.02, kd=0.02)


class MasterRight(Master):

    def __init__(self, dr):
        super().__init__(LEADERS_R, dr)

    def begin_for_operate(self):
        super().begin_for_operate()
        self.set_handle_zero()
        self.set_torque([0, 0.2, 0.1, 0, 0, 0])
        self.impedance_control(2, angle=0, speed=1, tff=0, kp=0.001, kd=0.02)
        # self.impedance_control(3, angle=0, speed=1, tff=0, kp=0.02, kd=0.02)


class PuppetRight(Puppet):

    def __init__(self, dr, gripper=None):
        super().__init__(FOLLOWERS_R, dr, gripper)

    def move_head_to_zero(self):
        self.dr.set_angle(self.id_list[-1],  0, speed=10, param=10, mode=1)
        self.dr.position_done(self.id_list[-1])


class PuppetLeft(Puppet):

    def __init__(self, dr):
        super().__init__(FOLLOWERS_L[:-1], dr)
        self.gripper_id = FOLLOWERS_L[-1]
        self.init_gripper()

    def init_gripper(self):
        self.dr.set_angle(self.gripper_id, 0, speed=10, param=10, mode=1)
        self.dr.position_done(self.gripper_id)
        logger.debug("gripper move to zero position")
        self.dr.set_angle_range(self.gripper_id, angle_min=FOLLOWERS_LEFT_ANGLE_MIN, angle_max=FOLLOWERS_LEFT_ANGLE_MAX)
        logger.debug(f"gripper set angle range {(FOLLOWERS_LEFT_ANGLE_MIN, FOLLOWERS_LEFT_ANGLE_MAX)}")

    def step_gripper(self, ratio: float):
        if ratio == 0:
            return
        # self.step(self.gripper_id, int(10 * ratio))
        self.dr.step_angle(self.gripper_id, int(20 * ratio), int(30 * ratio), 50, 0)
