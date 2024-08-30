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

from task_cube import task_cube_generate_text
from task_tea import task_tea_generate_text
from text_embedding import TextEmbedding, DATA_PATH, TextEmbeddingTransformer



def run():
    out = {}
    task_texts = task_cube_generate_text()
    for key, strs in task_texts.items():
        out[key] = embedding(strs)
    task_texts = task_tea_generate_text()
    for key, strs in task_texts.items():
        out[key] = embedding(strs)

    pickle.dump(out, open(DATA_PATH, 'wb'))
    print("save to ", DATA_PATH)
    return out


def embedding(texts) -> List[Tuple[str, np.ndarray]]:
    out = []
    for text in texts:
        out.append((text, text_emb.text2vec(text)))
    return out


if __name__ == '__main__':
    text_emb = TextEmbedding()
    run()
    tet = TextEmbeddingTransformer()
    tet.show()
