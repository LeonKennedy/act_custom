#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: camera.py
@time: 2024/4/11 11:15
@desc:
"""

import cv2
from constant import CAMERA_NAME, IMAGE_W, IMAGE_H


def check_camera():
    camera_indexes = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.read()[0]:
            print("find index", i)
            camera_indexes.append(i)
        else:
            print(i, "index not found")

        if cap.isOpened():
            cap.release()


def show_capture_info(cap):
    print("WIDTH", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "HEIGHT", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("TYPE", cap.get(cv2.CAP_PROP_FRAME_TYPE))


def show():
    caps = {}
    for name, id in CAMERA_NAME.items():
        cap = cv2.VideoCapture(id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(3, IMAGE_W)
        cap.set(4, IMAGE_H)
        show_capture_info(cap)
        caps[name] = cap

    imgs = {}
    ret = True
    for name, cap in caps.items():
        r, img = cap.read()
        if r:
            imgs[name] = img
            ret &= r
        else:
            print(name, "not found!", "ret:", r)
    while ret:
        for name, img in imgs.items():
            cv2.imshow(name, img)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break

        for name, cap in caps.items():
            r, img = cap.read()
            imgs[name] = img
            ret &= r

    for _, cap in caps.items():
        cap.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # check_camera()
    show()
