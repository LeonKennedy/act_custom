import time
import arm_robot as robot # 忽略这里的报错
uart_baudrate = 115200 # 串口波特率，与CAN模块的串口波特率一致，（出厂默认为 115200，最高460800）
com_master = 'COM11' # 在这里输入 COM 端口号（Windows 系统）
'''注意：主臂从下到上直到手爪电机 ID 号必须是依次 1,2,3,4,5,6,7'''
id_list_master = [1, 2, 3, 4, 5, 6, 7] # 设定从臂关节电机的 ID 号（含爪子）
id_list_slaver = [8, 9, 10, 11, 12, 13, 14] # 设定从臂关节电机的 ID 号（含爪子）
''''主臂参数'''''
arm_six_axes_l_p = 150 # 工具参考点到电机输出轴表面的距离，单位mm（所有尺寸参数皆为mm）
arm_six_axes_l_p_mass_center = 55 # 工具（负载）质心到 6 号关节输出面的距离
arm_six_axes_G_p = 0.396 # 负载重量，单位kg，所有重量单位皆为kg
''''主臂参数'''''

# # 主臂对象初始化函数函数
dr = robot.arm_robot(L_p=arm_six_axes_l_p, L_p_mass_center=arm_six_axes_l_p_mass_center, G_p=arm_six_axes_G_p, com=com_master, uart_baudrate=uart_baudrate)

''''主臂参数'''''
# robot.dimensions #
arm_six_axes_l_1 = 150   # 主臂l1杆的长度，单位mm
arm_six_axes_l_2 = 150   # 主臂l2杆的长度，单位mm
arm_six_axes_l_3 = 68.54   # 主臂l3杆的长度，单位mm
arm_six_axes_d_3 = 54.94   # 主臂d3杆的长度，单位mm
arm_six_axes_d_4 = 33   # 主臂d4杆的长度，单位mm
dr.L = [arm_six_axes_l_1, arm_six_axes_l_2, arm_six_axes_l_3, arm_six_axes_d_3, arm_six_axes_d_4 + arm_six_axes_l_p]
# # robot.weight #
arm_six_axes_G_1 = 0.0595  # 主臂杆件2重量，单位kg
arm_six_axes_G_2 = 0.329 # 主臂关节3重量，单位kg
arm_six_axes_G_3 = 0.0595 # 主臂杆件3重量，单位kg
arm_six_axes_G_4 = 0.468 # 主臂关节4+关节5重量，单位kg
arm_six_axes_G_5 = 0.199 # 主臂关节6，单位kg
dr.G = [arm_six_axes_G_1, arm_six_axes_G_2, arm_six_axes_G_3, arm_six_axes_G_4, arm_six_axes_G_5, arm_six_axes_G_p]
# # robot.joints_torque_factor #
arm_six_axes_joints_torque_factor_1 = 1 # 主臂关节1力矩修正系数
arm_six_axes_joints_torque_factor_2 = 1 # 主臂关节2力矩修正系数
arm_six_axes_joints_torque_factor_3 = 1 # 主臂关节3力矩修正系数
arm_six_axes_joints_torque_factor_4 = 1 # 主臂关节4力矩修正系数
arm_six_axes_joints_torque_factor_5 = 0.8 # 主臂关节5力矩修正系数
arm_six_axes_joints_torque_factor_6 = 1 # 主臂关节6力矩修正系数
dr.torque_factors = [arm_six_axes_joints_torque_factor_1, arm_six_axes_joints_torque_factor_2, arm_six_axes_joints_torque_factor_3, arm_six_axes_joints_torque_factor_4, arm_six_axes_joints_torque_factor_5, arm_six_axes_joints_torque_factor_6]
''''主臂参数'''''

'''先将两台臂运动到合适姿态（也可以在开机前摆到合适位置）'''
angles_init = [0, 15, -90, 0, 0, 0]
dr.set_angles(id_list=id_list_master[:6], angle_list=angles_init, speed=10, param=10, mode=1)
time.sleep(1)
dr.set_angles(id_list=id_list_slaver[:6], angle_list=angles_init, speed=10, param=10, mode=1)
time.sleep(5) # 等待运动到合适位置

dr.set_torque(id_num=id_list_master[6], torque=0, param=0, mode=0) # 放松主臂手抓电机
dr.set_torque_limit(id_num=id_list_slaver[6], torque_limit=0.3) # 设置从臂手爪的最大夹持力


'''开始主从操作'''
start = time.time()
N = 0 # 记录循环次数
t = 10 # 主从操作执行时间，单位s
id_list = id_list_master + id_list_slaver
global start_over
start_over = time.time()
while (time.time() - start < t):
    slaver_angle_list = []
    angle_speed_torque_list = dr.get_angle_speed_torque_all(id_list=id_list)
    if angle_speed_torque_list is None:
        pass
    else:

        for i in range(len(id_list_slaver)):
            slaver_angle_list.append(angle_speed_torque_list[i][0])
        print(slaver_angle_list)
        dr.gravity_compensation(pay_load=0, F=[0,0,0], angle_list=slaver_angle_list[:6])
        dr.set_angles(id_list=id_list_slaver, angle_list=slaver_angle_list, speed=20, param=15, mode=0)
        '''适当调整pid后可使用下面的代码'''
        # bit_wideth1 = 1 / (time.time() - start_over) / 2 # 计算在 t>n 情况下的指令发送频率的一半
        # dr.set_angles(id_list=id_list_slaver, angle_list=slaver_angle_list, speed=20, param=bit_wideth1, mode=0)
        # start_over = time.time()
        '''适当调整pid后可使用上面的代码'''
        time.sleep(0.001) # 不延时会卡住（即控制命令发送后不能立即回读数据）
        N += 1

#     if slaver_angle_list == []:
#         pass
#     else:
#         print(slaver_angle_list) # 打印角度
#         dr.set_angles(id_list=id_list_slaver, angle_list=slaver_angle_list, speed=10, param=10, mode=0)
#         i += 1
print(t, "s 内循环次数为：", N)
angles_ = []
for k in range(len(id_list_master)):
    angles_.append(dr.get_angle(id_list_master[k]))
dr.set_angles(id_list=id_list_master, angle_list=angles_, speed=10, param=10, mode=1) # 主臂急停在当前位姿