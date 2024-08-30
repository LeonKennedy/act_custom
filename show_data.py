import os.path
import pickle
import sys
import time

import cv2
import numpy as np


def show_image(name, bytes):
    buffer = np.frombuffer(bytes, np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    cv2.imshow(name, img)


def combine_image(images):
    b = np.concatenate([images["LEFT"], images['TOP'], images["RIGHT"]])
    return b


def show(data):
    print(data['task'])
    episodes = data['data']
    time.sleep((5))
    for episode in episodes:
        img = combine_image(episode["camera"])
        cv2.imshow("left-top-right", img)
        print("left master", episode["left_master"], "left puppet", episode["left_puppet"])
        print("right master", episode["right_master"], "right puppet", episode["right_puppet"])
        # print("trigger", episode["left_trigger"], episode["right_trigger"])
        print()
        key = cv2.waitKey(50)
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
