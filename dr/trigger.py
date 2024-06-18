import time
from loguru import logger
from pymodbus.client import ModbusSerialClient as ModbusClient


# 返回时间 50 旋转方向CCW
class Trigger:

    def __init__(self):
        self._s = ModbusClient(port='COM7', baudrate=9600)
        assert self._s.connected
        self.zero = self.raw_read()
        self.dead_zone = (self.zero - 20, self.zero + 20)
        self.range = TRIGGER_LEFT_RANGE

    def read(self) -> float:
        o = self.raw_read()
        if self.dead_zone[0] <= o <= self.dead_zone[1]:
            return 0
        logger.debug(f"read trigger raw data: {o}")

        if o > self.zero:
            return (o - self.zero) / (self.range[1] - self.zero)
        if o < self.zero:
            return (self.zero - o) / (self.range[0] - self.zero)

    def raw_read(self) -> int:
        o = self._s.read_holding_registers(0, 1, slave=1)
        if o.isError():
            print(f"Received Modbus library error({o})")
            return
        return o.registers[0]

    def read_continue(self):
        while 1:
            o = self.read()
            time.sleep(1 / 50)
            print(o)

    def __del__(self):
        self._s.close()


if __name__ == '__main__':
    # s = ModbusClient(method='rtu', port='COM7', baudrate=9600)
    t = Trigger()
