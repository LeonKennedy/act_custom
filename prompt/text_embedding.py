#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: text_embedding.py
@time: 2024/8/28 16:38
@desc:
"""
import os.path
import pickle
import random

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = "assets/task_embedding.pkl"


class TextEmbedding:
    def __init__(self):
        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

    def text2vec(self, text):
        return self.model.encode(text, normalize_embeddings=True)


class TextEmbeddingTransformer:

    def __init__(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.model = pickle.load(open(os.path.join(dir_path, DATA_PATH), 'rb'))

    def embedding(self, text) -> np.ndarray:
        elems =  self.model[text]
        return random.choice(elems)[1]