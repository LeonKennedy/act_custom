#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: action_chunk.py
@time: 2024/8/14 15:45
@desc:
"""

class ActionChunk:
    def __init__(self, chunk_size, fixed_length):
        self.chunk_size = chunk_size
        self.fixed_length = fixed_length