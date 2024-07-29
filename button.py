import time

import serial
import keyboard


class Button:

    def __init__(self, com_name, baud):
        self.timeout = 0.001
        self._s = serial.Serial(com_name, baudrate=baud, timeout=self.timeout)

    def block_waiting_press(self):
        time.sleep(0.1)
        self._s.reset_input_buffer()
        while 1:
            if 0 == self._s.in_waiting:
                time.sleep(self.timeout)
                continue
            else:
                break
        time.sleep(0.5)
        self._s.reset_input_buffer()

    def is_press(self) -> bool:
        return self._s.in_waiting > 0

    def reset_input_buffer(self):
        time.sleep(0.1)
        self._s.reset_input_buffer()

    @property
    def in_waiting(self):
        return self._s.in_waiting


running_flag = True


def _change_state(event):
    global running_flag
    running_flag = False
    print("in change")


def test_jiao_ta():
    print("Start?")
    keyboard.wait("5")
    keyboard.on_press_key("5", _change_state)
    i = 0
    while running_flag:
        i+=1
        time.sleep(i)
        print(f"waiting---{running_flag}")



if __name__ == '__main__':
    test_jiao_ta()
    # b = Button("COM6", 9600)
    # while 1:
    #     print("wait for start？")
    #     b.block_waiting_press()
    #
    #     while 1:
    #         print("doing...")
    #         time.sleep(1)
    #         if b.is_press():
    #             b.reset_input_buffer()
    #             print("save", b.in_waiting)
    #             break
    #
    # print("Done!", b.in_waiting)
