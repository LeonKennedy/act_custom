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
import time
import concurrent.futures
from typing import Dict
from tqdm.auto import tqdm

import cv2
import numpy as np

from dr.constants import CAMERA_NAME, IMAGE_W, IMAGE_H


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
        key = cv2.waitKey(20)
        if key == ord('q'):
            break

        for name, cap in caps.items():
            r, img = cap.read()
            imgs[name] = img
            ret &= r

    for _, cap in caps.items():
        cap.release()

    cv2.destroyAllWindows()


def _init_camera(name: str, i: int):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_H)
    assert cap.isOpened()
    print(name, cap.get(cv2.CAP_PROP_FOURCC), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap


class CameraGroup:

    def __init__(self):
        self.caps = {name: _init_camera(name, i) for name, i in CAMERA_NAME.items()}
        self.image_size = (IMAGE_H, IMAGE_W, 3)
        # self.tasks = {name: Thread(target=cap.read) for name, cap in self.caps.items()}

    def read_async(self) -> Dict[str, np.ndarray]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exc:
            futures = {name: exc.submit(cap.read) for name, cap in self.caps.items()}
            results = {}
            for name, future in futures.items():
                ret, img = future.result()
                results[name] = img
            return results

    def read_sync(self) -> Dict[str, np.ndarray]:
        results = {}
        for name, cap in self.caps.items():
            ret, img = cap.read()
            assert ret, f"{name} error"
            if name in ("LEFT", "RIGHT"):
                results[name] = cv2.resize(img, (IMAGE_W, IMAGE_H))
            else:
                results[name] = img
        return results

    def read_stack(self) -> np.ndarray:
        imgs = self.read_sync()
        img = np.stack([imgs['TOP'], imgs["LEFT"], imgs["RIGHT"]])
        return img

    def show(self):
        while 1:
            imgs = self.read_sync()
            for name, img in imgs.items():
                cv2.imshow(name, img)
            key = cv2.waitKey(10)
            if key == ord('q'):
                break

    def read_once(self):
        pass


def test_async_with_sync(cnt=100):
    print("test start!")
    start = time.time()
    for _ in tqdm(range(cnt)):
        out = cg.read_sync()
    print("SYNC time:", round(time.time() - start))

    start = time.time()
    for _ in tqdm(range(cnt)):
        # out = cg.caps['TOP'].read()
        out = cg.caps['FRONT'].read()
    print("ASYNC time:", round(time.time() - start))

    start = time.time()
    for _ in tqdm(range(cnt)):
        out = cg.caps['TOP'].read()
        out = cg.caps['FRONT'].read()
    print("ASYNC time:", round(time.time() - start))


if __name__ == '__main__':
    # check_camera()
    # show()
    cg = CameraGroup()
    # cg.read_stack()
    # test_async_with_sync()
    cg.show()
