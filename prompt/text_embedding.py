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

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-zh-v1.5')


def text2vec(text):
    return model.encode(text, normalize_embeddings=True)
