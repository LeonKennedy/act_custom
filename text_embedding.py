#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: text_embedding.py
@time: 2024/8/27 18:23
@desc:
"""


# sentences_1 = ["样例数据-1", "样例数据-2"]
# sentences_2 = ["样例数据-3", "样例数据-4"]

# embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
# embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
#
# print(embeddings_1)
# print(embeddings_2)
# print(embeddings_1.shape, embeddings_2.shape)
# similarity = embeddings_1 @ embeddings_2.T
# print(similarity)


def text2vec(text):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    return model.encode(text, normalize_embeddings=True)


def task_assemble():
    i = input("use arm?\n1. left\n2. right")
    arm = 'left' if i == '1' else 'right'
    j = input("which color of cube?\n1. red\n2. blue\n3. yellow")
    cube_color = 'red' if j == '1' else 'blue' if j == '2' else 'yellow'
    k = input("which color of box?\n1. red\n2. blue")
    box_color = 'red' if k == '1' else 'blue'
    return arm, cube_color, box_color


if __name__ == '__main__':
    print(task_assemble())
