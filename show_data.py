import os.path
import pickle
import sys

import cv2
import numpy as np


def show_image(name, bytes):
    buffer = np.frombuffer(bytes, np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    cv2.imshow(name, img)


def show(episodes):
    for episode in episodes:
        show_image("top", episode["top"])
        show_image("right", episode["right"])
        print("puppet", episode["right_puppet"])
        print("master", episode["right_master"])
        print("gripper", episode["right_gripper"])
        print()
        key = cv2.waitKey(10)
        if key == ord('q'):
            break


if __name__ == '__main__':
    p = sys.argv[1]
    if os.path.exists(p):
        with open(p, "rb") as f:
            data = pickle.load(f)
            show(data)
    else:
        print("Error: not found!")
