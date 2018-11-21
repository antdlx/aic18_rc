# -*- coding: utf-8 -*-
import argparse

from QANet.preprocess import process_data


def run_preprocess(in_path,out_path):

    dataRoot = './data/pickles'
    data = dataRoot
    threshold = 1
    embed_dim = 300
    pretrained_embedding_path = dataRoot+'jwe_word2vec_size300.txt'
    train_path = 'aic18/train2_dot_com.json'
    valid_path = 'aic18/dev2_dot_com.json'
    testa_path = 'aic18/aic_test.json'
    out_embedding_path = dataRoot+'embedding.table'
    out_word2id_path = dataRoot+'word2id.table'

    # train_path 是输入路径，valid_path是输出路径
    vocab_size = process_data(data, in_path,out_path,testa_path,
                              threshold,embed_dim,pretrained_embedding_path,
                              out_embedding_path,out_word2id_path)

