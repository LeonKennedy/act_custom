from typing import List, Optional
from .DrEmpower_can import DrEmpower_can
from .gripper import Grasper
from .constants import LEADERS_R, FOLLOWERS_R


class Robot:
    def __init__(self, id_list: List, dr: DrEmpower_can, gripper: Grasper = None):
        self.id_list = id_list
        self.dr = dr
        if gripper:
            self.gripper = gripper

        self.recording = False

    def set_torque(self, torque_list):
        self.dr.disable_angle_speed_torque_state(id_num=0)  # 先全部取消状态反馈

        self.dr.set_torques(id_list=self.id_list, torque_list=torque_list, param=3, mode=0)

    def impedance_control(self, id_num=0, angle=0, speed=0, tff=0, kp=0, kd=0):
        self.dr.impedance_control(id_num, angle=angle, speed=speed, tff=tff, kp=kp, kd=kd)

    def set_zero(self, sid: Optional[int] = None):
        if sid is None:
            for id in self.id_list:
                self.dr.set_zero_position(id)
        else:
            self.dr.set_zero_position(sid)

    def step(self, sid: int, angle: int):
        self.dr.step_angle(sid, angle, speed=10, param=10, mode=1)

    def set_angle(self, sid: int, angle: int):
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

    # ---         gripper   --

    @property
    def gripper_status(self):
        return self.gripper.status


class Master(Robot):

    def set_handle_zero(self):
        print("ser", self.id_list[-1], "set to zero position")
        self.dr.set_zero_position(self.id_list[-1])

    def begin_for_operate(self):
        self.set_handle_zero()
        self.set_torque([0, 0.2, 0.1, 0, 0, 0])
        self.impedance_control(2, angle=0, speed=1, tff=0, kp=0.001, kd=0.02)


class Puppet(Robot):

    def open_gripper(self, event=None):
        self.gripper.loose()

    def close_gripper(self, event=None):
        self.gripper.clamp()

    def change_gripper(self, event=None):
        self.gripper.change()
        print("gripper status", self.gripper_status)

    def set_gripper(self, gripper):
        if gripper == 1:
            self.open_gripper()
        elif gripper == 2:
            self.close_gripper()


class MasterRight(Master):

    def __init__(self, dr):
        super().__init__(LEADERS_R, dr)


class PuppetRight(Puppet):

    def __init__(self, dr, gripper=None):
        super().__init__(FOLLOWERS_R, dr, gripper)
