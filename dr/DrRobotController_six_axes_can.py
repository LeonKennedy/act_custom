# -*- coding=utf-8 -*-

"""
大然机器人-六轴机械臂标准模型控制器函数库

适用平台：windows、linux（jetson nano、树莓派）
编程语言：python 3
通信硬件：USB 转 CAN 模块

****************************机械臂参数设置函数*************************
                          write_param：设置机械臂参数。
                    save_robot_config：保存机械臂参数
                 robot_ instantiation：实例化机器人

****************************运动控制函数*************************
                         set_arm_pose：控制机械臂末端运动到指定位置和姿态函数
                     set_arm_position：控制机械臂末端运动到指定位置函数
                        set_arm_P_Y_R：控制机械臂末端运动到指定姿态函数
                set_arm_relative_pose：控制机械臂末端运动到相对当前的指定位置和姿态函数
            set_arm_relative_position：控制机械臂末端运动到相对当前的指定位置函数
               set_arm_relative_P_Y_R：控制机械臂末端运动到相对当前的指定姿态函数
                          robot_estop：控制机械臂急停函数
                            pose_done：等待机械臂运动到位函数
              set_arm_poses_curve_pre：预设机械臂末端轨迹函数
               set_arm_poses_curve_do：机械臂末端轨迹执行函数

****************************示教编程函数*************************
                             add_pose：位姿示教编程函数
                              do_pose：位姿示教执行函数
                     tutorial_program：轨迹示教编程函数
                          tutorial_do：轨迹示教执行函数

****************************力控功能函数*************************
             impedance_control_joints：关节阻抗函数
                      zero_force_pull：零力拖动函数

****************************参数回读函数************************
                      show_pose_x_y_z：查看内存中机械臂当前位置函数
                      show_pose_P_Y_R：查看内存中机械臂当前姿态函数
                show_joint_z_position：查看内存中机械臂各关节的 z 坐标函数
                show_joint_x_position：查看内存中机械臂各关节在机械臂所在平面内的 x 坐标函数
                    detect_pose_x_y_z：查看现实中机械臂当前位置函数
                    detect_pose_P_Y_R：查看现实中机械臂当前姿态函数
              detect_joint_z_position：查看现实中机械臂各关节的 z 坐标函数
              detect_joint_x_position：查看现实中机械臂各关节在机械臂所在平面内的 x 坐标函数
                           read_param：查看机械臂参数
            read_joint_motor_property：查看关节电机参数

****************************关节电机参数设置函数************************
           write_joint_motor_property：设置关节电机参数
            motor_control_save_config：保存关节电机参数
      motor_control_set_zero_position：设置关节电机零点位置函数

****************************机械臂辅助功能************************
                                 free：机械臂待机（各关节卸力）函数
                                 lock：机械臂自锁函数

****************************关节电机控制功能************************
              motor_control_set_angle：单个关节电机绝对角度控制函数
             motor_control_step_angle：单个关节电机相对角度控制函数
     motor_control_set_angle_adaptive：单个关节电机力位混合（自适应）绝对角度控制函数
      motor_control_impedance_control：单个关节电机阻抗控制函数
             motor_control_motion_aid：单个关节电机运动助力函数
                        position_done：等待单个关节电机运动到位函数
              motor_control_set_speed：单个关节电机转速控制函数
             motor_control_set_torque：单个关节电机力矩控制函数
     motor_control_set_speed_adaptive：设置单个关节电机力位混合（自适应）转速函数
    motor_control_set_torque_adaptive：设置单个关节电机力位混合（自适应）力矩函数
                  motor_control_estop：关节电机急停函数

"""

import sys
import time
import serial
from serial.serialutil import SerialException

from parameter_interface import *
import math as cm
import struct

uart_baudrate = 115200  # 串口波特率，与CAN模块的串口波特率一致，（出厂默认为 115200，最高460800）

try:
    uart = serial.Serial('COM5', uart_baudrate)  # 在 windows 下控制一体化关节，相应的输入连接的COM口和波特率
except SerialException as e:
    print("WARNNING COM5 OPEN ERROR")

READ_FLAG = 0  # 读取结果标志位

POINTS_NUM = 1000

"""
功能函数，用户使用
"""


def write_param(id_num=1, property='', value=0):
    '''
    可设置机械臂各杆件尺寸、各部分重量等参数。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param property: 参数名称，详见 parameter_interface.py
    :param value: 参数值
    :return: True or False
    '''
    try:
        data_types = {'f': 0, 'u16': 1, 's16': 2, 'u32': 3, 's32': 4}  # 字典
        address = property
        if type(property) == str:  # 判断属性名称是否为字符
            address = key_find_value(property)  # 返回属性对应的编码
            if address > 0:
                pass
                # print(str(value_find_key(address)) + ' address = ' + str(address))
            else:
                print('invalid property: ' + str(property))
                return False
        data_type = property_type.get(value_find_key(address))  # 确定属性的数据类型
        # print(data_type)
        data = format_data([address, data_types.get(data_type), value], 'u16 u16 ' + data_type, 'encode')
        send_command(id_num=id_num, cmd=0x0D, data=data, rtr=0)  # 需要用标准帧（数据帧）进行发送，不能用远程帧
    except Exception as e:
        print("---error in write_property---：", e)
        return False
    return True


def save_robot_config(id_num=1):
    '''
    可保存机械臂的各杆件尺寸、重量等参数（注意该函数不保存关节电机的参数）。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :return: 无
    '''
    data = format_data([1], 'u32', 'encode')
    send_command(id_num=id_num, cmd=0x0F, data=data, rtr=0)


def robot_instantiation(id_num=1):
    '''
    应在保存机械臂参数后使用。该函数执行后控制器内部将实例化机械臂，以便使用其他库函数控制机械臂。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :return: 无
    '''
    data = format_data([0], 'u32', 'encode')
    send_command(id_num=id_num, cmd=0x1B, data=data, rtr=0)


def set_arm_pose(id_num=1, pl_temp=[0, 0, 0], theta_P_Y_R=[0, 0, 0], speed=1, acceleration=10):
    """
    控制机械臂末端运动到全局坐标系 中指定位置和姿态。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param pl_temp: 设置机械臂末端坐标系原点在O- x0y0z0中的坐标(x, y, z)，单位mm
    :param theta_P_Y_R: 设置机械臂末端坐标系在O- x0y0z0中的姿态角(Pitch, Yaw, Roll)，单位°
    :param speed: 设置关节电机最大转动速度，单位（r/min）
    :param acceleration: 设置各关节电机角加速度（(r/min)/s）
    :return: 无
    """
    preset_8(id_num=id_num, value_1=pl_temp[0], value_2=pl_temp[1], value_3=pl_temp[2], value_4=theta_P_Y_R[0],
             value_5=theta_P_Y_R[1], value_6=theta_P_Y_R[2], value_7=speed, value_8=acceleration)
    data = format_data([1, 1], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x0A, data=data, rtr=0)


def set_arm_position(id_num=1, pl_temp=[0, 0, 0], speed=1, acceleration=10):
    '''
    控制机械臂末端运动到全局坐标系 中指定位置，而不改变末端姿态。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param pl_temp: 设置机械臂末端坐标系原点在O- x0y0z0中的坐标(x, y, z)，单位mm
    :param speed: 设置关节电机最大转动速度，单位（r/min）
    :param acceleration: 设置各关节电机角加速度（(r/min)/s）
    :return: 无
    '''
    preset_5(id_num=id_num, value_1=pl_temp[0], value_2=pl_temp[1], value_3=pl_temp[2], value_4=speed,
             value_5=acceleration)
    data = format_data([1, 2], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x0A, data=data, rtr=0)


def set_arm_P_Y_R(id_num=1, theta_P_Y_R=[0, 0, 0], speed=1, acceleration=10):
    '''
    控制机械臂末端运动到全局坐标系 中指定姿态，而不改变末端位置。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param theta_P_Y_R: 设置机械臂末端坐标系在O- x0y0z0中的姿态角(Pitch, Yaw, Roll)，单位°
    :param speed: 设置关节电机最大转动速度，单位（r/min）
    :param acceleration: 设置各关节电机角加速度（(r/min)/s）
    :return: 无
    '''
    preset_5(id_num=id_num, value_1=theta_P_Y_R[0], value_2=theta_P_Y_R[1], value_3=theta_P_Y_R[2], value_4=speed,
             value_5=acceleration)
    data = format_data([1, 3], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x0A, data=data, rtr=0)


def set_arm_relative_pose(id_num=1, pl_temp=[0, 0, 0], theta_P_Y_R=[0, 0, 0], speed=1, acceleration=10):
    '''
    控制机械臂末端相对于当前位置和姿态在全局坐标系O- x0y0z0中运动到指定位置和姿态。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param pl_temp: 设置机械臂末端坐标系原点在O- x0y0z0中的相对坐标(x, y, z)，单位mm
    :param theta_P_Y_R: 设置机械臂末端坐标系在O- x0y0z0中的相对姿态角(Pitch, Yaw, Roll，如图2-1)，单位°
    :param speed: 设置关节电机最大转动速度，单位（r/min）
    :param acceleration: 设置各关节电机角加速度（(r/min)/s）
    :return: 无
    '''
    preset_8(id_num=id_num, value_1=pl_temp[0], value_2=pl_temp[1], value_3=pl_temp[2], value_4=theta_P_Y_R[0],
             value_5=theta_P_Y_R[1], value_6=theta_P_Y_R[2], value_7=speed, value_8=acceleration)
    data = format_data([1, 4], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x0A, data=data, rtr=0)


def set_arm_relative_position(id_num=1, pl_temp=[0, 0, 0], speed=1, acceleration=10):
    '''
    控制机械臂末端相对于当前位置在全局坐标系O- x0y0z0中运动到指定位置，而不改变末端姿态。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param pl_temp: 设置机械臂末端坐标系原点在O- x0y0z0中的相对坐标(x, y, z)，单位mm
    :param speed: 设置关节电机最大转动速度，单位（r/min）
    :param acceleration: 设置各关节电机角加速度（(r/min)/s）
    :return:
    '''
    preset_5(id_num=id_num, value_1=pl_temp[0], value_2=pl_temp[1], value_3=pl_temp[2], value_4=speed,
             value_5=acceleration)
    data = format_data([1, 5], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x0A, data=data, rtr=0)


def set_arm_relative_P_Y_R(id_num=1, theta_P_Y_R=[0, 0, 0], speed=1, acceleration=10):
    '''
    控制机械臂末端相对于当前姿态在全局坐标系O- x0y0z0中运动到指定姿态，而不改变末端位置。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param theta_P_Y_R: 设置机械臂末端坐标系在O- x0y0z0中的相对姿态角(Pitch, Yaw, Roll)，单位
    :param speed: 设置关节电机最大转动速度，单位（r/min）
    :param acceleration: 设置各关节电机角加速度（(r/min)/s）
    :return: 无
    '''
    preset_5(id_num=id_num, value_1=theta_P_Y_R[0], value_2=theta_P_Y_R[1], value_3=theta_P_Y_R[2], value_4=speed,
             value_5=acceleration)
    data = format_data([1, 6], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x0A, data=data, rtr=0)


def robot_estop(id_num=1):
    '''
    直接控制机械臂停止运动（执行轨迹规划、轨迹示教除外）。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :return: 无
    '''
    for i in range(6):
        motor_control_estop(id_num=id_num, joint_num=i + 1)


def pose_done(id_num=1):
    '''
    该函数运行后将阻塞程序执行直到机械臂运动到上一命令指定的位姿。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :return: 无
    '''
    data = format_data([0, 7], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x0A, data=data, rtr=0)
    while read_param(id_num=1, property='dr.robot.pose_done') == 0:
        pass


def set_arm_poses_curve_pre(id_num=1, pls_temp=[], theta_P_Y_Rs_temp=[]):
    '''
    提前记录末端轨迹点（末端在全局坐标系O- x0y0z0中的位置和姿态），并将其发送至控制器，等待执行。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param pls_temp: 机械臂末端轨迹上截取的多个点在O- x0y0z0中的坐标(x, y, z)所组成的列表，单位mm
    :param theta_P_Y_Rs_temp: 机械臂末端轨迹上截取的多个点在O- x0y0z0中的多个姿态角(Pitch, Yaw, Roll)组成的列表，单位°
    :return: 无
    '''
    if len(pls_temp) > len(theta_P_Y_Rs_temp):
        for ii in range(len(pls_temp) - len(theta_P_Y_Rs_temp)):
            theta_P_Y_Rs_temp.append(
                theta_P_Y_Rs_temp[len(theta_P_Y_Rs_temp) - 1])  # 如果输入的姿态角数目少于坐标值数目，则缺少的姿态角用最后一组姿态角补足
    if len(theta_P_Y_Rs_temp) > len(pls_temp):
        for ii in range(len(theta_P_Y_Rs_temp) - len(pls_temp)):
            pls_temp.append(pls_temp[len(pls_temp) - 1])  # 如果输入的坐标值数目少于姿态角数目，则缺少的姿态角用最后一组坐标值补足
    n = len(pls_temp)
    print(n)
    for i in range(n):
        if i > POINTS_NUM - 1:
            return
        pl_temp = pls_temp[i]
        theta_P_Y_R_temp = theta_P_Y_Rs_temp[i]
        print(pl_temp + theta_P_Y_R_temp)
        preset_6(id_num=id_num, value_1=pl_temp[0], value_2=pl_temp[1], value_3=pl_temp[2], value_4=theta_P_Y_R_temp[0],
                 value_5=theta_P_Y_R_temp[1], value_6=theta_P_Y_R_temp[2])
        time.sleep(0.001)
        if i == POINTS_NUM - 1 or i == n - 1:
            data = format_data([1], 'u32', 'encode')  # 1 表示发送完成
            send_command(id_num=id_num, cmd=0x1F, data=data, rtr=0)
        else:
            data = format_data([0], 'u32', 'encode')  # 0 表示发送未完成
            send_command(id_num=id_num, cmd=0x1F, data=data, rtr=0)
        time.sleep(0.001)


def set_arm_poses_curve_do(id_num=1, t=1):
    '''
    将预设机械臂末端轨迹函数set_arm_poses_curve_pre()记录末端轨迹点取出，并在指定时间（大致时间）内执行完毕。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param t: 轨迹执行时间
    :return:
    '''
    data = format_data([t], 'f', 'encode')
    send_command(id_num=id_num, cmd=0x01, data=data, rtr=0)


def add_pose(id_num=1, t=1):
    '''
    用于将机械臂当前位姿和对应的保持时间记录到系统内存的姿态列表中。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param t: 当前姿态在示教复现时保持的时间，单位s
    :return: 无
    '''
    data = format_data([t], 'f', 'encode')
    send_command(id_num=id_num, cmd=0x1C, data=data, rtr=0)


def do_pose(id_num=1, speed=10, acceleration=10, o_r=0, n=0):
    '''
    用于执行通过位姿示教保存到位姿列表中的位姿。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param speed: 关节电机最大转速，单位r/min
    :param acceleration: 关节电机角加速度（(r/min)/s）
    :param o_r: 用来选择执行的顺序，o_r =0：从前往后执行（默认）, o_r =1: 从后往前执行
    :param n: 用来控制执行的细节，如果n=0（默认），这执行所有保存的姿态，如果n>0 则执行第n个姿态
    :return: 无
    '''
    preset_4(id_num=id_num, value_1=speed, value_2=acceleration, value_3=0, value_4=0)
    data = format_data([o_r, n], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x1D, data=data, rtr=0)


def tutorial_program(id_num=1, pay_load=0, F=[0, 0, 0], n=1000, t=5):
    '''
    用于示教一条轨迹。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param pay_load: 末端负载，单位kg，需据实填写，若负载非零请确定负载执行到末端关节端面的距离
    :param F: 末端在全局坐标系坐标轴方向上的受力，单位kg
    :param n: 轨迹中点的数量（最大 1000）
    :param t: 轨迹示教的时长
    :return: 无
    '''
    preset_4(id_num=id_num, value_1=pay_load, value_2=F[0], value_3=F[1], value_4=F[2])
    data = format_data([n, t], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x19, data=data, rtr=0)


def tutorial_do(id_num=1, t=5):
    '''
    执行示教后得到的轨迹。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param t: 轨迹执行的大致耗时
    :return: 无
    '''
    data = format_data([t], 'f', 'encode')
    send_command(id_num=id_num, cmd=0x1A, data=data, rtr=0)


def impedance_control_joints(id_num=1, kp=1, kd=1):
    '''
    用于设定在当前位姿下各关节阻抗系数。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param kp: 刚度系数，单位：度/Nm
    :param kd: 阻尼系数，单位r/min/Nm
    :return: 无
    '''
    preset_4(id_num=id_num, value_1=0, value_2=0, value_3=kp, value_4=kd)
    data = format_data([0], 'u32', 'encode')
    send_command(id_num=id_num, cmd=0x18, data=data, rtr=0)


def gravity_compensation(id_num=1, pay_load=0):
    '''
    进入机械臂处于重力平衡状态，循环发送该指令可实现零力拖动
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param pay_load: 末端负载，单位kg，需据实填写，若负载非零请确定负载执行到末端关节端面的距离
    :return:
    '''
    data = format_data([pay_load, 1], 'f u16', 'encode')
    send_command(id_num=id_num, cmd=0x17, data=data, rtr=0)
    time.sleep(0.01)


def out_of_gravity_compensation(id_num=1):
    '''
    退出机械臂处于重力平衡状态。（版本号240528以上版本可用）
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :return:
    '''
    data = format_data([0, 0], 'f u16', 'encode')
    send_command(id_num=id_num, cmd=0x17, data=data, rtr=0)


def zero_force_pull(id_num=1, pay_load=0):
    '''
    效果为使用很小的外力即可拖动机械臂，并且机械臂处于重力平衡状态。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param pay_load: 末端负载，单位kg，需据实填写，若负载非零请确定负载执行到末端关节端面的距离
    :return:
    '''
    data = format_data([pay_load, 255], 'f u16', 'encode')
    send_command(id_num=id_num, cmd=0x17, data=data, rtr=0)


def show_pose_x_y_z(id_num=1):
    '''
    可显示内存中机械臂末端的位置和姿态。（版本号240528以上版本可用）
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :return: 内存中机械臂末端位置[x, y, z]
    '''
    data = format_data([0, 1], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x0B, data=data, rtr=0)
    x_y_z = []
    for i in range(3):
        cdata = receive_data()
        if READ_FLAG == 1:
            # property = format_data(data=cdata, format='u16 u16 ' + 'f', type='decode')
            x_y_z.append(format_data(data=cdata, format='u16 u16 ' + 'f', type='decode')[-1])
            if len(x_y_z) == 0:
                print("参数错误")
                return False
    return x_y_z


def show_pose_P_Y_R(id_num=1):
    '''
    可显示内存中机械臂末端的姿态。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :return: 内存中机械臂末端姿态 [pitch, yaw, roll]
    '''
    data = format_data([0, 2], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x0B, data=data, rtr=0)
    P_Y_R = []
    for i in range(3):
        cdata = receive_data()
        if READ_FLAG == 1:
            # property = format_data(data=cdata, format='u16 u16 ' + 'f', type='decode')
            P_Y_R.append(format_data(data=cdata, format='u16 u16 ' + 'f', type='decode')[-1])
            if len(P_Y_R) == 0:
                print("参数错误")
                return False
    return P_Y_R


def show_joint_z_position(id_num=1, n=0):
    '''
    可显示内存中当前位姿下机械臂各关节在全局坐标系下的z坐标，并返回指定编号运动关节的z坐标（编号范围为1~6）。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param n: 关节编号
    :return: 第n号关节在全局坐标系中的z坐标
    '''
    data = format_data([n, 3], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x0B, data=data, rtr=0)
    cdata = receive_data()
    if READ_FLAG == 1:
        property = format_data(data=cdata, format='u16 u16 ' + 'f', type='decode')
        if len(property) == 0:
            print("参数错误")
            return False
        return property[-1]


def show_joint_x_position(id_num=1, n=0):
    '''
    可显示内存中当前位姿下机械臂各关节在机械臂所在平面内的x坐标，并返回指定编号运动关节的x坐标（编号范围为1~6）。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param n: 关节编号
    :return: 第n号关节在全局坐标系中的z坐标
    '''
    data = format_data([n, 4], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x0B, data=data, rtr=0)
    cdata = receive_data()
    if READ_FLAG == 1:
        property = format_data(data=cdata, format='u16 u16 ' + 'f', type='decode')
        if len(property) == 0:
            print("参数错误")
            return False
        return property[-1]


def detect_joints(id_num=1, n=1):
    '''
    可显示现实中机械臂指定编号的关节模型角度。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param n: 关节编号
    :return: 指定编号关节模型角度
    '''
    data = format_data([n, 5], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x0B, data=data, rtr=0)
    cdata = receive_data()
    if READ_FLAG == 1:
        property = format_data(data=cdata, format='u16 u16 ' + 'f', type='decode')
        if len(property) == 0:
            print("参数错误")
            return False
        return property[-1]


def detect_pose_x_y_z(id_num=1):
    '''
    可显示现实中机械臂末端的位置。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :return: 现实中机械臂末端位置和姿态
    '''
    data = format_data([0, 6], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x0B, data=data, rtr=0)
    x_y_z = []
    for i in range(3):
        cdata = receive_data()
        if READ_FLAG == 1:
            # property = format_data(data=cdata, format='u16 u16 ' + 'f', type='decode')
            x_y_z.append(format_data(data=cdata, format='u16 u16 ' + 'f', type='decode')[-1])
            if len(x_y_z) == 0:
                print("参数错误")
                return False
    return x_y_z


def detect_pose_P_Y_R(id_num=1):
    '''
    可显示现实中机械臂末端姿态。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :return: 现实中机械臂末端姿态
    '''
    data = format_data([0, 7], 'u32 u32', 'encode')
    send_command(id_num=id_num, cmd=0x0B, data=data, rtr=0)
    P_Y_R = []
    for i in range(3):
        cdata = receive_data()
        if READ_FLAG == 1:
            # property = format_data(data=cdata, format='u16 u16 ' + 'f', type='decode')
            P_Y_R.append(format_data(data=cdata, format='u16 u16 ' + 'f', type='decode')[-1])
            if len(P_Y_R) == 0:
                print("参数错误")
                return False
    return P_Y_R


def read_param(id_num=0, property=''):
    '''
    可查看机械臂各部分尺寸、重量等参数。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param property: 参数名称，详见 parameter_interface.py
    :return:
    '''
    try:
        data_types = {'f': 0, 'u16': 1, 's16': 2, 'u32': 3, 's32': 4}
        address = property
        if type(property) == str:
            address = key_find_value(property)
            if address > 0:
                pass
            else:
                print('invalid property: ' + str(property))
                return False
        data_type = property_type.get(value_find_key(address))
        data = format_data([address, data_types.get(data_type)], 'u16 u16', 'encode')
        send_command(id_num=id_num, cmd=0x0C, data=data, rtr=0)  # 需要用标准帧（数据帧）进行发送，不能用远程帧
        cdata = receive_data()
        if READ_FLAG == 1:
            property = format_data(data=cdata, format='u16 u16 ' + data_type, type='decode')
            if len(property) > 0:
                return property[-1]
            else:
                print("参数错误")
                return False
        else:
            print("参数读取失败")
            return False
    except Exception as e:
        print("---error in read_property---：", e)
        return False


def read_joint_motor_property(id_num=1, joint_num=1, property=''):
    '''
    可查看对应编号关节的参数（编号范围为1~6）。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param joint_num: 关节编号
    :param property: 参数名称，详见 parameter_interface.py
    :return:
    '''
    try:
        data_types = {'f': 0, 'u16': 1, 's16': 2, 'u32': 3, 's32': 4}
        address = property
        if type(property) == str:
            address = key_find_value(property)
            if address > 0:
                pass
            else:
                print('invalid property: ' + str(property))
                return False
        data_type = property_type.get(value_find_key(address))
        data = format_data([address, data_types.get(data_type), joint_num], 'u16 u16 u16', 'encode')
        send_command(id_num=id_num, cmd=0x0C, data=data, rtr=0)  # 需要用标准帧（数据帧）进行发送，不能用远程帧
        cdata = receive_data()
        if READ_FLAG == 1:
            property = format_data(data=cdata, format='u16 u16 ' + data_type, type='decode')
            if len(property) > 0:
                return property[-1]
            else:
                print("参数错误")
                return False
        else:
            print("参数读取失败")
            return False
    except Exception as e:
        print("---error in read_property---：", e)
        return False


def write_joint_motor_property(id_num=1, joint_num=1, property='', value=0):
    '''
    可设置对应编号关节电机的参数（编号范围为1~6）。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param joint_num: 需要设置参数的关节编号
    :param property: 参数名称，详见 parameter_interface.py
    :param value: 参数值
    :return:
    '''
    preset_4(id_num=id_num, value_1=joint_num, value_2=1)
    write_param(id_num=id_num, property=property, value=value)


def motor_control_save_config(id_num=1, joint_num=1):
    '''
    保存关节电机属性参数。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param joint_num: 关节编号
    :return:
    '''
    data = format_data([joint_num, 1, 11], 'u32 u16 u16', 'encode')
    send_command(id_num=id_num, cmd=0x0E, data=data, rtr=0)


def motor_control_set_zero_position(id_num=1, joint_num=1):
    '''
    用于将关节电机当前位置设置为角度0位。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param joint_num: 关节编号
    :return: 无
    '''
    data = format_data([joint_num, 1, 12], 'u32 u16 u16', 'encode')
    send_command(id_num=id_num, cmd=0x0E, data=data, rtr=0)


def free(id_num=1):
    '''
    可使机械臂各关节卸力，以便手动摆放机械臂位姿，或待机以降低功耗。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :return: 无
    '''
    data = format_data([1], 'u32', 'encode')
    send_command(id_num=id_num, cmd=0x1E, data=data, rtr=0)


def lock(id_num=1):
    '''
    可使机械臂各关节从待机状态恢复自锁。常在位姿示教编程结束后使用。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :return: 无
    '''
    data = format_data([2], 'u32', 'encode')
    send_command(id_num=id_num, cmd=0x1E, data=data, rtr=0)


def motor_control_set_angle(id_num=1, joint_num=1, angle=0, speed=0, param=0, mode=0):
    '''
    用于控制指定编号的关节电机按照指定的转速转动到指定的角度（绝对角度，相对于用户设定的零点角度）。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param joint_num: 关节编号
    :param angle: 关节电机绝对角度（°）。
    :param speed: 关节电机转速（r/min），具体含义由 mode 的取值决定，mode=1：目标转速；mode=0/2：前馈转速。
    :param param: 运动参数，由 mode 取值决定，mode=0：角度输入滤波带宽（<300）；mode=1：启动和停止阶段角加转速（r/min/s）；mode=2：前馈力矩 torque（Nm）。
    :param mode: 角度控制模，关节电机支持三种角度控制模式，由 mode 取值决定，
                mode = 0：轨迹跟踪模式，适合多个轨迹点输入后进行平滑控制，角度输入滤波带宽参数需设置为指令发送频率的一半；
                mode = 1：梯形轨迹模式，这种模式下可以指定运动过程中的目标转速和启停加转速；
                mode = 2：前馈控制模式，这种模式下的 speed 和 torque 分别为前馈控制量。前馈控制在原有PID控制基础上加入转速和力矩前馈，提高系统的响应特性和减少静态误差。
    :return: 无
    :note: 注：在 mode=1 梯形轨迹模式中，speed 和 accel 都要大于 0；mode=0 时 speed 不起作用。
    '''
    preset_4(id_num=id_num, value_1=angle, value_2=speed, value_3=param)
    data = format_data([joint_num, mode, 1], 'u32 u16 u16', 'encode')
    send_command(id_num=id_num, cmd=0x0E, data=data, rtr=0)


def motor_control_step_angle(id_num=1, joint_num=1, angle=0, speed=0, param=0, mode=0):
    '''
    控制指定编号的关节电机按照指定的转速相对转动指定的角度（相对角度，相对于发送该指令时的角度）。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param joint_num: 关节编号
    :param angle: 关节电机相对角度（°）
    :param speed: 关节电机转速（r/min），具体含义由 mode 的取值决定，mode=1：目标转速；mode=0/2：前馈转速。
    :param param: 运动参数，由 mode 取值决定，mode=0：角度输入滤波带宽（<300）；mode=1：启动和停止阶段角加转速（r/min/s）；mode=2：前馈力矩 torque（Nm）。
    :param mode: 角度控制模，关节电机支持三种角度控制模式，由 mode 取值决定，
                mode = 0：轨迹跟踪模式，适合多个轨迹点输入后进行平滑控制，角度输入滤波带宽参数需设置为指令发送频率的一半；
                mode = 1：梯形轨迹模式，这种模式下可以指定运动过程中的目标转速和启停加转速；
                mode = 2：前馈控制模式，这种模式下的 speed 和 torque 分别为前馈控制量。前馈控制在原有PID控制基础上加入转速和力矩前馈，提高系统的响应特性和减少静态误差。
    :return: 无
    :note: 注：在 mode=1 梯形轨迹模式中，speed 和 accel 都要大于 0；mode=0 时 speed 不起作用。
    '''
    preset_4(id_num=id_num, value_1=angle, value_2=speed, value_3=param)
    data = format_data([joint_num, mode, 2], 'u32 u16 u16', 'encode')
    send_command(id_num=id_num, cmd=0x0E, data=data, rtr=0)


def motor_control_set_angle_adaptive(id_num=1, joint_num=1, angle=0, speed=0, torque=0):
    '''
    控制指定编号的关节电机按照限定的转速和力矩转动到指定的角度（绝对角度，相对于用户设定的零点角度）。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param joint_num: 关节编号
    :param angle: 关节电机角度（°）。
    :param speed: 限定转速值（r/min）。
    :param torque: 限定力矩值（Nm)。
    :return: 无
    '''
    preset_4(id_num=id_num, value_1=angle, value_2=speed, value_3=torque)
    data = format_data([joint_num, 1, 3], 'u32 u16 u16', 'encode')
    send_command(id_num=id_num, cmd=0x0E, data=data, rtr=0)


def motor_control_impedance_control(id_num=1, joint_num=1, angle=0, speed=0, tff=0, kp=0, kd=0):
    '''
    对指定编号的关节电机进行阻抗控制。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param joint_num: 关节编号
    :param angle: 关节电机目标角度（°）。
    :param speed: 关节电机目标转速（r/min）。
    :param tff: 前馈力矩（Nm)。
    :param kp: 角度刚度系数（Nm/°），需大于 0。
    :param kd: 转速阻尼系数（Nm/(r/min)），需大于 0。
    :return: 无
    :note: (1)	该函数直接控制关节输出力矩，其目标输出力矩计算公式如下：
                torque = kp*( angle – angle_) + tff + kd*(speed – speed_)
                其中 angle_ 和 speed_ 分别为输出轴当前实际角度（°）和当前实际转速（r/min）, kp 和 kd 为刚度系数和阻尼系数。
           (2)	默认关节在机器人不能连续整周旋转，因而该函数内部强制kp不能为零，如需在连续整周转的场景使用该函数，可自行修改改函数体对应代码
    '''
    preset_5(id_num=id_num, value_1=angle, value_2=speed, value_3=tff, value_4=kp, value_5=kd)
    data = format_data([joint_num, 1, 4], 'u32 u16 u16', 'encode')
    send_command(id_num=id_num, cmd=0x0E, data=data, rtr=0)


def motor_control_motion_aid(id_num=1, joint_num=1, angle=0, speed=0, angle_err=0, speed_err=0, torque=0):
    '''
    指定编号的关节电机进行运动助力。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param joint_num: 关节编号
    :param angle: 助力目标角度（°），该值减去关节当前角度即为助力行程。
    :param speed: 限定转速（r/min），即助力的限定转速，防止助力力矩引起的加速导致转速过快
    :param angle_err: 角度差值（°），表示运动助力的角度灵敏度。
    :param speed_err: 转速差值（r/min），表示运动助力的转速灵敏度。
    :param torque: 助力力矩（Nm)。
    :return: 无
    :note: a、当助力与外部驱动力之和大于阻力，关节会持续转动；
           b、当助力与外部驱动力之和小于阻力，关键开始减速，当转速小于 2 倍转速差值 speed_err 时，关节停止输出助力；
           c、一般情况下，该功能为人进行助力，强烈建议用户将助力力矩设置在人力所能及的范围内，即人力可使关节停止转动；
           d、若必须设置超出人力的力矩，则必须在合理位置设置牢固的机械限位，以避免超出运动范围给人或物体带来损伤。
    '''
    preset_5(id_num=id_num, value_1=angle, value_2=speed, value_3=angle_err, value_4=speed_err, value_5=torque)
    data = format_data([joint_num, 1, 5], 'u32 u16 u16', 'encode')
    send_command(id_num=id_num, cmd=0x0E, data=data, rtr=0)


def motor_control_position_done(id_num=1, joint_num=1):
    '''
    等待关节电机转动到指定角度。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param joint_num: 关节编号
    :return: 无
    '''
    while read_joint_motor_property(id_num=id_num, joint_num=joint_num, property='dr.controller.position_done') == 0:
        pass


def motor_control_set_speed(id_num=1, joint_num=1, speed=10, param=1, mode=1):
    '''
    控制指定编号的关节电机按照指定的转速连续整周转动（转动到关节支持的极限角度后自动停止）。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param joint_num: 关节编号
    :param speed: 目标转速（r/min）。
    :param param: 运动参数，由 mode 取值决定：mode=0，前馈力矩（Nm)；mode=1，目标加转速（r/min/s）。
    :param mode: 控制模式选择，由 mode 取值决定：
                 mode=0，转速直接控制模式，将关节电机目标转速直接设为 speed；
                 mode=1，匀加速控制模式，关节电机将按照目标角加速变化到 speed。
    :return: 无
    '''
    preset_4(id_num=id_num, value_1=speed, value_2=param)
    data = format_data([joint_num, mode, 6], 'u32 u16 u16', 'encode')
    send_command(id_num=id_num, cmd=0x0E, data=data, rtr=0)


def motor_control_set_torque(id_num=1, joint_num=1, torque=1, param=1, mode=1):
    '''
    控制指定 ID 编号的关节电机输出指定的力矩（Nm），若阻力不足以抵抗该力矩，则关节会持续转动（转动到关节支持的极限角度后自动停止）。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param joint_num: 关节编号
    :param torque: 目标力矩（Nm）
    :param param: 运动参数，由 mode 取值决定：
    :param mode: 控制模式选择，由 mode 取值决定：
                 mode=0，力矩直接控制模式，将关节电机目标转速直接设为 torque
                 mode=1，力矩匀速增加模式，关节电机将按照指定的单位时间内的增量匀速变化到 torque。
    :return: 无
    '''
    preset_4(id_num=id_num, value_1=torque, value_2=param)
    data = format_data([joint_num, mode, 7], 'u32 u16 u16', 'encode')
    send_command(id_num=id_num, cmd=0x0E, data=data, rtr=0)


def motor_control_set_speed_adaptive(id_num=1, joint_num=1, speed_adaptive=1):
    '''
    设置关节电机力位混合（自适应）转速限制 speed_adaptive （r/min），此后关节力位混合（自适应）转速绝对值不超过 speed_adaptive。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param joint_num: 关节编号
    :param speed_adaptive: 力位混合（自适应）转速（r/min）（必须大于 0）。
    :return: 无
    :note: 本函数需在运动指令之后运行。
    '''
    preset_4(id_num=id_num, value_1=speed_adaptive)
    data = format_data([joint_num, 1, 8], 'u32 u16 u16', 'encode')
    send_command(id_num=id_num, cmd=0x0E, data=data, rtr=0)


def motor_control_set_torque_adaptive(id_num=1, joint_num=1, torque_adaptive=1):
    '''
    设置关节电机力位混合（自适应）力矩限制 torque_adaptive (Nm)，此后关节力位混合（自适应）转速绝对值不超过 torque_adaptive。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param joint_num: 关节编号
    :param torque_adaptive: 力位混合（自适应）力矩（Nm）（必须大于 0）。
    :return: 无
    :note: 本函数需在运动指令之后运行。
    '''
    preset_4(id_num=id_num, value_1=torque_adaptive)
    data = format_data([joint_num, 1, 9], 'u32 u16 u16', 'encode')
    send_command(id_num=id_num, cmd=0x0E, data=data, rtr=0)


def motor_control_estop(id_num=1, joint_num=1):
    '''
    控制关节电机紧急停止。
    :param id_num: 机械臂 ID 号，如果不知道 ID 号则写0，但此时需保证总线中只有一台机械臂
    :param joint_num: 关节编号
    :return:
    '''
    data = format_data([joint_num, 1, 10], 'u32 u16 u16', 'encode')
    send_command(id_num=id_num, cmd=0x0E, data=data, rtr=0)


"""
内部辅助函数，用户无需使用
"""


# 串口发送函数
def write_data(data=[]):
    global uart
    try:
        result = uart.write(data)  # 写数据
        # print("write_data: ", data)
        return result
    except Exception as e:
        print("---error in write_data--：", e)
        print("重启串口")
        uart.close()
        uart.open()
        result = uart.write(data)  # 写数据
        return result


# 串口接收函数
def read_data(num=16):  # 16个字节
    global READ_FLAG
    READ_FLAG = -1
    byte_list = []
    byte_list_head = 0
    i = 500  # 经过测试，发现正常接收16字节耗时大概为500
    while uart.inWaiting() == 0 and i > 0:  # To do:
        i -= 1
        time.sleep(0.01)
        if i == 0:
            print("数据为空，接收数据超时，程序退出")
            sys.exit()
    while byte_list_head != 170:
        byte_list_head = uart.read(1)[0]
    byte_list.append(byte_list_head)
    while uart.inWaiting() > 0 and len(byte_list) < num:
        byte_list.append(list(uart.read(1))[0])
    if len(byte_list) == num:
        READ_FLAG = 1
        return byte_list
    elif len(byte_list) > num:
        byte_list_length = len(byte_list)
        READ_FLAG = 1
        for i in range(byte_list_length - num):
            byte_list.pop(byte_list_length - i - 1)
        return byte_list
    else:
        print("Received data error in read_data(): " + str(byte_list))
        READ_FLAG = -1
        byte_list = []
        return


def read_data_state(n):  # 16*n个字节
    global READ_FLAG
    READ_FLAG = -1
    byte_list = []
    byte_list_head = 0
    i = 500  # 经过测试，发现正常接收16字节耗时大概为500
    while uart.inWaiting() == 0 and i > 0:  # To do:
        i -= 1
        time.sleep(0.001)
        if i == 0:
            print("数据为空，接收数据超时，程序退出")
            sys.exit()
    while byte_list_head != 170:
        byte_list_head = uart.read(1)[0]
    byte_list.append(byte_list_head)
    while uart.inWaiting() > 0 or len(byte_list) < (n * 16):
        byte_list.append(list(uart.read(1))[0])
    if len(byte_list) == (n * 16):
        READ_FLAG = 1
        return byte_list
    elif len(byte_list) > (n * 16):
        byte_list_length = len(byte_list)
        READ_FLAG = 1
        for i in range(byte_list_length - (n * 16)):
            byte_list.pop(byte_list_length - i - 1)
        return byte_list
    else:
        print("Received data error in read_data_state(): " + str(byte_list))
        READ_FLAG = -1
        return


def read_data_state2(n):  # 16*n个字节
    global READ_FLAG
    READ_FLAG = -1
    byte_list = []
    byte_list_head = 0
    while uart.inWaiting() == 0:
        pass
    while byte_list_head != 170:
        byte_list_head = uart.read(1)[0]
    byte_list.append(byte_list_head)
    while uart.inWaiting() > 0 or len(byte_list) < (n * 16):
        byte_list.append(list(uart.read(1))[0])
    if len(byte_list) == (n * 16):
        READ_FLAG = 1
        return byte_list
    elif len(byte_list) > (n * 16):
        byte_list_length = len(byte_list)
        READ_FLAG = 1
        for i in range(byte_list_length - (n * 16)):
            byte_list.pop(byte_list_length - i - 1)
        return byte_list
    else:
        print("Received data error in read_data_state(): " + str(byte_list))
        READ_FLAG = -1
        return


def read_data_id():  # 16*n个字节
    byte_list = []
    id_list = []
    i = 500  # 经过测试，发现正常接收16字节耗时大概为500
    while uart.inWaiting() == 0 and i > 0:  # To do:
        i -= 1
        time.sleep(0.001)
        if i == 0:
            print("数据为空，接收数据超时，程序退出")
            sys.exit()
    while uart.inWaiting() > 0:
        byte_list.append(list(uart.read(1))[0])
    if len(byte_list) % 16 == 0 and len(byte_list) >= 16:
        for i in range(len(byte_list) // 16):
            jdata = byte_list[i * 16: (i + 1) * 16]
            cdata = uart_to_can_ID(data=jdata)
            id_list.append((cdata[1] * 256 + cdata[2] - 1) >> 5)
        return id_list
    else:
        print("Received data error in read_data_id(): " + str(byte_list))
        return


# USB转CAN模块包模式：CAN报文->串行帧
def can_to_uart(data=[], rtr=0):
    udata = [0xAA, 0, 0, 0x08, 0, 0, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]  # 其中0x08代表CAN读取字头中的DLC（报文长度）
    # udata[1]对应CAN字头中的IDE，udata[2]对应CAN字头中的RTR，udata[3]对应CAN字头中的DLC，udata[4~7]对应id
    # udata 根据USB转CAN模块串口包模式定义
    if len(data) == 11 and data[0] == 0x08:  # 0x08 为预设的一个校验字节
        if rtr == 1:
            udata[2] = 0x01  # rtr 标志位
        for i in range(10):
            udata[6 + i] = data[i + 1]
        return udata
    else:
        return []


# USB转CAN模块包模式：串行帧->CAN报文
def uart_to_can(data=[]):
    global READ_FLAG
    cdata = [0x08, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    # if len(data) == 16 and data[3] == 0x08:
    if (len(data) % 16 == 0) and data[3] == 0x08:
        for i in range(10):
            cdata[1 + i] = data[i + 6]
        return cdata
    else:
        READ_FLAG = -1
        return []


# USB转CAN模块包模式：串行帧->CAN报文，同时计算 CAN 节点 ID
def uart_to_can_ID(data=[]):
    global READ_FLAG
    cdata = [0x08, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    if len(data) == 16 and data[3] == 0x08:
        for i in range(10):
            cdata[1 + i] = data[i + 6]
        return cdata
    else:
        READ_FLAG = -1
        # print(READ_FLAG)
        return [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]


# CAN发送函数
def send_command(id_num=0, cmd=0x09, data=[], rtr=0):
    global set_angles_mode_1_flag
    cdata = [0x08, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    id_list = (id_num << 5) + cmd  # id号右移5位 + cmd，此处根据odrive CAN node_id号与cmd_id的角度关系决定
    cdata[1] = id_list >> 8
    cdata[2] = id_list & 0xFF
    for i in range(8):
        cdata[3 + i] = data[i]  # data[]中包含命令的内容，如角度、转速、转矩等
    # print("cdata: ", cdata)
    data = can_to_uart(data=cdata, rtr=rtr)
    # print("send_command: ", data)
    write_data(data)
    set_angles_mode_1_flag = 0
    # write_data(data=can_to_uart(data=cdata, rtr=rtr))


# CAN接收函数
def receive_data():
    udata = read_data(16)  # can 报文一共16个字节
    if READ_FLAG == 1:
        cdata = uart_to_can(data=udata)
        return cdata[3:]


def format_data(data=[], format="f f", type='decode'):
    # print("format_data data:", data)
    # print("format_data format:", format)
    format_list = format.split()  # 将数据类型转换成列表
    rdata = []
    if type == 'decode' and len(data) == 8:
        p = 0
        for f in format_list:
            s_f = []
            if f == 'f':
                s_f = [4, 'f']
            elif f == 'u16':
                s_f = [2, 'H']
            elif f == 's16':
                s_f = [2, 'h']
            elif f == 'u32':
                s_f = [4, 'I']
            elif f == 's32':
                s_f = [4, 'i']
            ba = bytearray()
            if len(s_f) == 2:
                for i in range(s_f[0]):
                    ba.append(data[p])
                    p = p + 1
                rdata.append(struct.unpack(s_f[1], ba)[0])
            else:
                print('unkown format in format_data(): ' + f)
                return []
        return rdata
    elif type == 'encode' and len(format_list) == len(data):  # 判断数据格式类型数量与数据数量相同
        for i in range(len(format_list)):
            f = format_list[i]
            s_f = []
            if f == 'f':
                s_f = [4, 'f']  # f 代表float，占4个字节, angle 是浮点数
            elif f == 'u16':
                s_f = [2, 'H']  # H 代表unsigned short，占2个字节
            elif f == 's16':
                s_f = [2, 'h']  # h 代表short，占2个字节，speed 和 加转速是短整型
            elif f == 'u32':
                s_f = [4, 'I']  # I 代表 unsigned int, 占4个字节
            elif f == 's32':
                s_f = [4, 'i']  # i 代表int，占4个字节
            if len(s_f) == 2:
                bs = struct.pack(s_f[1], data[i])  # 将数据转换成二进制数
                for j in range(s_f[0]):
                    rdata.append(bs[j])  # 将数据按字节装进 rdara
                    # print(i, ":", rdata)
            else:
                print('unkown format in format_data(): ' + f)
                return []
        if len(rdata) < 8:
            for i in range(8 - len(rdata)):
                rdata.append(0x00)
        return rdata


def preset_4(id_num=1, value_1=0, value_2=0, value_3=0, value_4=0):
    data = format_data([value_1, value_2], 'f f', 'encode')
    send_command(id_num=id_num, cmd=0x04, data=data, rtr=0)
    data = format_data([value_3, value_4], 'f f', 'encode')
    send_command(id_num=id_num, cmd=0x04, data=data, rtr=0)


def preset_5(id_num=1, value_1=0, value_2=0, value_3=0, value_4=0, value_5=0):
    data = format_data([value_1, value_2], 'f f', 'encode')
    send_command(id_num=id_num, cmd=0x05, data=data, rtr=0)
    data = format_data([value_3, value_4], 'f f', 'encode')
    send_command(id_num=id_num, cmd=0x05, data=data, rtr=0)
    data = format_data([value_5], 'f', 'encode')
    send_command(id_num=id_num, cmd=0x05, data=data, rtr=0)


def preset_6(id_num=1, value_1=0, value_2=0, value_3=0, value_4=0, value_5=0, value_6=0):
    data = format_data([value_1, value_2], 'f f', 'encode')
    send_command(id_num=id_num, cmd=0x06, data=data, rtr=0)
    data = format_data([value_3, value_4], 'f f', 'encode')
    send_command(id_num=id_num, cmd=0x06, data=data, rtr=0)
    data = format_data([value_5, value_6], 'f f', 'encode')
    send_command(id_num=id_num, cmd=0x06, data=data, rtr=0)


def preset_8(id_num=1, value_1=0, value_2=0, value_3=0, value_4=0, value_5=0, value_6=0, value_7=0, value_8=0):
    data = format_data([value_1, value_2], 'f f', 'encode')
    send_command(id_num=id_num, cmd=0x08, data=data, rtr=0)
    data = format_data([value_3, value_4], 'f f', 'encode')
    send_command(id_num=id_num, cmd=0x08, data=data, rtr=0)
    data = format_data([value_5, value_6], 'f f', 'encode')
    send_command(id_num=id_num, cmd=0x08, data=data, rtr=0)
    data = format_data([value_7, value_8], 'f f', 'encode')
    send_command(id_num=id_num, cmd=0x08, data=data, rtr=0)


def preset_9(id_num=1, value_1=0, value_2=0, value_3=0, value_4=0, value_5=0, value_6=0, value_7=0, value_8=0,
             value_9=0):
    data = format_data([value_1, value_2], 'f f', 'encode')
    send_command(id_num=id_num, cmd=0x09, data=data, rtr=0)
    data = format_data([value_3, value_4], 'f f', 'encode')
    send_command(id_num=id_num, cmd=0x09, data=data, rtr=0)
    data = format_data([value_5, value_6], 'f f', 'encode')
    send_command(id_num=id_num, cmd=0x09, data=data, rtr=0)
    data = format_data([value_7, value_8], 'f f', 'encode')
    send_command(id_num=id_num, cmd=0x09, data=data, rtr=0)
    data = format_data([value_9], 'f', 'encode')
    send_command(id_num=id_num, cmd=0x09, data=data, rtr=0)
