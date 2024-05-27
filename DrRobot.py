import pickle
# import math
# import time
import DrEmpower_can as Dr # 忽略此处的报错
# # dr = Dr.DrEmpower_can(com='/dev/ttyAMA0', uart_baudrate=baudrate) # 在树莓派（raspbian）下控制一体化关节，相应的输入连接的串口
# # dr = Dr.DrEmpower_can(com='/dev/ttyUSB0', uart_baudrate=baudrate) # 在 jetson nano（ubuntu） 下控制一体化关节，相应的输入连接的串口
# # dr = Dr.DrEmpower_can(com='/dev/cu.usbserial-110', uart_baudrate=baudrate) # 在苹果电脑 mac 下控制一体化关节，相应地输入串口
# import DrEmpower_can as dr ### 如果用的是面向过程编程就这样
import math
import time
import serial
import keyboard



class Robot:
    def __init__(self, id_list, dr, gripper=None, gripper_id=0):
        id_num = 2
        self.id_list = id_list
        self.dr = dr
        self.FPS = 50
        if gripper:
            self.gripper = gripper
            self.gripper_id = gripper_id
            self.gripper_status = -1
            self.open_gripper()

        self.recording = False


# exit()
# dr.set_mode(id_num, 2
#             )
# dr.set_torque(id_num=id_num, torque=0, param=1, mode=1)

    def set_torque(self, torque_list):
        self.dr.disable_angle_speed_torque_state(id_num=0)  # 先全部取消状态反馈

        self.dr.set_torques(id_list=self.id_list, torque_list=torque_list, param=3, mode=0)

    def impedance_control(self, id_num=0, angle=0, speed=0, tff=0, kp=0, kd=0):
        self.dr.impedance_control(id_num, angle=angle, speed=speed, tff=tff, kp=kp, kd=kd)

    def set_zero(self):
        for id in self.id_list:
            self.dr.set_zero_position(id)
    def move_zero(self):
        self.dr.set_angles(id_list=self.id_list, angle_list=[0, 0, 0, 0, 0, 0], speed=10, param=10, mode=1)  # 先控制关节已梯形估计模型平缓运动到曲线起点

    def move_to(self, angle_list, wait = False):
        print('move_to:', angle_list)
        self.dr.set_angles(id_list=self.id_list, angle_list=angle_list, speed=10, param=10, mode=1)  # 先控制关节已梯形估计模型平缓运动到曲线起点
        if wait:
            self.dr.positions_done(self.id_list)

    def move_to2(self, angle_list, bit_width):
        self.dr.set_angles(id_list=self.id_list, angle_list=angle_list, speed=20, param=bit_width, mode=0)

    def open_gripper(self, event=None):
        if self.gripper_status != 1:
            if self.gripper_id == 2:
                pos = "0800"
            elif self.gripper_id == 1:
                pos = 1000
            write_len = self.gripper.write(f'#00{self.gripper_id}P{pos}T0500!'.encode('utf-8'))
            self.gripper_status = 1

    def close_gripper(self, event=None):
        if self.gripper_status != 2:
            if self.gripper_id == 2:
                pos = 1400
            elif self.gripper_id == 1:
                pos = 1850
            write_len = self.gripper.write(f'#00{self.gripper_id}P{pos}T0500!'.encode('utf-8'))
            self.gripper_status = 2
    def change_gripper(self, event = None):
        # print('change_gripper:')
        # if self.gripper_status != 1:
        self.open_gripper()
        # elif self.gripper_status != 2:
        self.close_gripper()
    def set_gripper(self, gripper):
        if gripper == 1:
            self.open_gripper()
        elif gripper == 2:
            self.close_gripper()

    def get_angles(self):
        angles = []
        begin = time.time()
        for id in self.id_list:
            angles.append(self.dr.get_angle(id))
        print('get:', (time.time() - begin))
        return angles



