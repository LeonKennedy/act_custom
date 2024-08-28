#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: prepare_data.py
@time: 2024/8/28 16:36
@desc:
"""
from typing import List, Tuple

import numpy as np
import pickle

from task_select import task_cube_generate_text
from text_embedding import text2vec

data_path = "assets/task_embedding.pkl"


def run():
    task_texts = task_cube_generate_text()
    out = {}
    for key, strs in task_texts.items():
        out[key] = embedding(strs)

    pickle.dump(out, open(data_path, 'wb'))
    print("save to ", data_path)
    return out


def embedding(texts) -> List[Tuple[str, np.ndarray]]:
    out = []
    for text in texts:
        out.append((text, text2vec(text)))
    return out


if __name__ == '__main__':
    run()
