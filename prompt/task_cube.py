#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: task_select.py
@time: 2024/8/28 16:25
@desc:
"""
from typing import Tuple, Dict, List
from itertools import product

TASK_1 = (('left', 'right'), ('red', 'blue', 'yellow'), ('red', 'blue'))


def _input_string(keys: Tuple):
    out = ''
    for i, name in zip(range(1, len(keys) + 1), keys):
        out += f"{i}. {name}"
    return out


def cube_assemble():
    i = input(f"use arm? {_input_string(TASK_1[0])}\n")
    arm = TASK_1[0][int(i) - 1]
    j = input(f"which color of cube? {_input_string(TASK_1[1])}\n")
    cube_color = TASK_1[1][int(j) - 1]
    k = input(f"which color of box? {_input_string(TASK_1[2])}\n")
    box_color = TASK_1[2][int(k) - 1]
    return arm, cube_color, box_color


# def _task_list():
#     arm, cube_color, box_color =
#     out = []
#     for i, j, k in zip(arm, cube_color, box_color):
#         out.append((i, j, k))
#     return out

def elem_to_text(elem: Tuple) -> Dict[Tuple, List[str]]:
    arm, cube_color, box_color = elem
    sens = []
    sens.append(f"use {arm} arm take {cube_color} cube, then put into {box_color} box")
    return {elem: sens}


def task_cube_generate_text() -> Dict[Tuple, List[str]]:
    task_list = product(*TASK_1)
    out = {}
    for e in task_list:
        out.update(elem_to_text(e))
    return out


if __name__ == '__main__':
    print(task_cube_generate_text())