import msvcrt

from .scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS


def getch():
    return msvcrt.getch().decode()


class Feite:
    MAX_ANGLE = 3400
    MIN_ANGLE = 0

    def __init__(self, sid: int, port_handler: PortHandler):
        self.sid = sid
        self.port_handler = port_handler
        self.packet_handler = sms_sts(self.port_handler)
        if self.port_handler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            getch()
            quit()

    def ping(self):
        scs_model_number, scs_comm_result, scs_error = self.packet_handler.ping(self.sid)
        if scs_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(scs_comm_result))
        else:
            print("[ID:%03d] ping Succeeded. SCServo model number : %d" % (self.sid, scs_model_number))
        if scs_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(scs_error))

    def read_angle(self) -> int:
        scs_model_number, scs_comm_result, scs_error = self.packet_handler.ReadPos(self.sid)
        if scs_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(scs_comm_result))
        else:
            return scs_model_number
        if scs_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(scs_error))


class Grasper(Feite):
    def set_angle(self, angle: float, speed: int = 4000, acc: int = 50):
        limit_angle = max(min(self.MAX_ANGLE, angle), self.MIN_ANGLE)
        scs_comm_result, scs_error = self.packet_handler.WritePosEx(self.sid, int(limit_angle), speed, acc=acc)
        if scs_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(scs_comm_result))
        elif scs_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(scs_error))

    def set_angle_by_ratio(self, ratio: float):
        angle = (self.MAX_ANGLE - self.MIN_ANGLE) * ratio
        self.set_angle(angle)

    def set_torque_limit(self, val: int):
        scs_comm_result, scs_error = self.packet_handler.setTorqueLimit(self.sid, val)
        if scs_comm_result != COMM_SUCCESS:
            print("%s" % self.packet_handler.getTxRxResult(scs_comm_result))
        elif scs_error != 0:
            print("%s" % self.packet_handler.getRxPacketError(scs_error))


def build_grasper():
    port_handler = PortHandler("COM11", 1_000_000)
    return Grasper(2, port_handler), Grasper(1, port_handler)
