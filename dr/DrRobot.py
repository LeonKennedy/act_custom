import time
from typing import List, Optional
from loguru import logger

from .DrEmpower_can import DrEmpower_can
from .trigger.dx_trigger import Trigger, build_trigger
from .constants import FOLLOWERS_R, FOLLOWERS_L, TRIGGER_NAME, LEADERS_R, LEADERS_L


# from . import DrRobotController_six_axes_can as arm_dr


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
        print('servo', self.id_list, 'move_to:', angle_list)
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

    def free(self):
        self.set_torque([0, 0, 0, 0, 0, 0])

    def set_zeros(self):
        for i in self.id_list:
            self.dr.set_zero_position(i)
        logger.debug(f"{self.__class__.__name__} set {self.id_list} zero position!")

    # ---         gripper   --


class ModuleMaster:

    def __init__(self, adr, arm_id: int, trigger: Trigger):
        self.adr = adr
        self.arm_id = arm_id
        self.trigger = trigger

    def free(self):
        self.adr.free(self.arm_id)

    def lock(self):
        self.adr.lock(self.arm_id)

    def gravity(self):
        self.adr.gravity_compensation(self.arm_id, 0)

    def out_gravity(self):
        self.adr.out_of_gravity_compensation(self.arm_id)

    def get_angle(self, i: int):
        return self.adr.read_joint_motor_property(self.arm_id, i, 'dr.output_shaft.angle')

    def set_angle(self, i: int, angle: float):
        self.adr.motor_control_set_angle(id_num=self.arm_id, joint_num=i, angle=angle, speed=10, param=10, mode=1)

    def step_angle(self, i: int, angle: float):
        self.adr.motor_control_step_angle(id_num=self.arm_id, joint_num=i, angle=angle, speed=10, param=10, mode=10)

    def read_angles(self):
        servo_angle_list = []
        for i in range(1, 7):
            servo_angle_list.append(self.adr.read_joint_motor_property(self.arm_id, i, 'dr.output_shaft.angle'))
        return servo_angle_list

    def set_zeros(self):
        for i in range(1, 7):
            self.adr.motor_control_set_zero_position(self.arm_id, i)
            time.sleep(1)
            self.adr.motor_control_save_config(self.arm_id, i)
            print("set", i, "done.")
            time.sleep(2)
        self.adr.robot_instantiation(self.arm_id)


class Puppet(Robot):

    def __init__(self, id_list: List, dr: DrEmpower_can):
        super().__init__(id_list, dr)
        # self.gripper_id = id_list[-1]
        self.gripper_current_angle = None
        # self.angle_range = (0, GRIPPER_RANGE_MAX)
        # self.init_gripper()

    def _check_is_have(self, sid: int) -> bool:
        return True
        # if sid == self.gripper_id:
        #     return True
        # return super()._check_is_have(sid)

    def init_gripper(self):
        self.dr.set_angle(self.gripper_id, 150, speed=10, param=10, mode=1)
        self.gripper_current_angle = 150
        # self.dr.position_done(self.gripper_id)

        # self.dr.set_angle_range(self.gripper_id, angle_min=self.angle_range[0], angle_max=self.angle_range[1])
        # logger.debug(f"{self.__class__.__name__} set angle range {self.angle_range}")

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

    def free(self):
        zero_torque_list = [0] * len(self.id_list)
        self.dr.set_torques(self.id_list, zero_torque_list, mode=0)

    @property
    def gripper_status(self):
        return self.gripper.status


class Master(Robot):

    def __init__(self, id_list: List, dr: DrEmpower_can):
        super().__init__(id_list, dr)


class MasterLeft(Master):
    pass


class MasterRight(Master):
    def __init__(self, dr: DrEmpower_can):
        super().__init__(LEADERS_R, dr)


class PuppetRight(Puppet):

    def __init__(self, dr):
        super().__init__(FOLLOWERS_R, dr)


class PuppetLeft(Puppet):

    def __init__(self, dr):
        super().__init__(FOLLOWERS_L, dr)


def build_master():
    left_trigger, right_trigger = build_trigger(TRIGGER_NAME)
    ml = MasterLeft(arm_dr, 2, left_trigger)
    mr = MasterRight(arm_dr, 1, right_trigger)
    return ml, mr


def build_puppet(dr):
    return PuppetLeft(dr), PuppetRight(dr)


def build_left(dr):
    pass


def build_right(dr):
    mr = MasterRight(dr)
    pr = PuppetRight(dr)
    return mr, pr


def build_arm(dr):
    master_right, puppet_right = build_right(dr)
    # puppet_left, puppet_right = build_puppet(dr)
    # master_left, master_right = build_master()
    return None, None, master_right, puppet_right


def get_all_angle(dr):
    servo_list = LEADERS_L + LEADERS_R + FOLLOWERS_L + FOLLOWERS_R
    sorted_servos = sorted(servo_list)
    while 1:
        state = dr.get_angle_speed_torque_all(sorted_servos)
        if state is None:
            logger.warning("read state error!")
            time.sleep(0.001)
            continue
        angle = [s[0] for s in state]
        if state:
            master_left = _get_value_by_index(angle, LEADERS_L)
            master_right = _get_value_by_index(angle, LEADERS_R)
            puppet_left = _get_value_by_index(angle, FOLLOWERS_L)
            puppet_right = _get_value_by_index(angle, FOLLOWERS_R)
            return master_left, puppet_left, master_right, puppet_right


def _get_value_by_index(angle, ind) -> List:
    out = []
    for i in ind:
        try:
            out.append(angle[i - 1])
        except IndexError as e:
            continue
    return out
