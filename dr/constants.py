import sys

# COM_NAME = 'COM3' # 在 windows 下控制一体化关节，相应的输入连接的COM口和波特率
# COM_NAME = '/dev/ttyAMA0' # 在树莓派（raspbian）下控制一体化关节，相应的输入连接的串口
# COM_NAME ='/dev/ttyUSB0' # 在 jetson nano（ubuntu） 下控制一体化关节，相应的输入连接的串口
# COM_NAME = "/dev/cu.usbmodem7C968E8F06521"
if sys.platform == "darwin":
    COM_NAME = "/dev/tty.usbmodem7C968E8F06521"
    GRASPER_NAME = '/dev/tty.wchusbserial14330'
    CAMERA_NAME = {"TOP": 1, "RIGHT": 0}
else:
    COM_NAME = "COM3"
    GRASPER_NAME = 'COM8'
    CAMERA_TOP = 1
    CAMERA_RIGHT = 0
    TRIGGER_NAME = "COM10"
    CAMERA_NAME = {"TOP": 1, "RIGHT": 0, "FRONT": 2, "LEFT": 3}
    BUTTOM_NAME = "COM11"

BAUDRATE = 115200  # 串口波特率，与CAN模块的串口波特率一致，（出厂默认为 115200，最高460800）

# LEADERS_R = [1, 2, 3, 4, 5, 6]
# LEADERS_L = [13, 14, 15, 16, 17, 18]
FOLLOWERS_R = [1, 2, 3, 4, 5, 6, 7]
FOLLOWERS_L = [8, 9, 10, 11, 12, 13, 14]

# GRIPPER_RIGHT_ID = 1
# TRIGGER_LEFT_RANGE = (186, 862)
GRIPPER_RANGE_MAX = 210

IMAGE_W = 640
IMAGE_H = 360

TRIGGER_MOTOR_ID_LEFT = 2
TRIGGER_MOTOR_ID_RIGHT = 4

FPS = 8
