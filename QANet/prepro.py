import json
import pickle

import jieba
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from QANet.run_preprocess import run_preprocess


def GetEmbeddingFromNoHeadTXT(file_name,word_dim=100):
    # vocab = []
    embedding_table = []
    word2id = {}
    cnt = 0
    fr = open(file_name, 'r',encoding='utf8',errors='ignore')

    # vocab.append("unk")
    word2id['unk'] = 0
    embedding_table.append([0.]*word_dim)

    for line in fr:
        row = line.strip().split(' ')
        if len(row[1:]) == word_dim:
            cnt += 1
            word2id[row[0]] = cnt
            # vocab.append(row[0])
            embedding_table.append(np.array(row[1:],dtype=np.float32).tolist())
            if cnt % 10000 == 0:
                print("loading w2v :current is {0}".format(cnt))
    print("vocab_size is {0}, embed_table size is {1}".format(len(word2id),len(embedding_table)))
    print("loaded word2vec")
    fr.close()
    # return vocab, embedding_table, word2id
    return  embedding_table, word2id

def ANSEstablishTFRecordsFromJSON_VER20(word2id,input_file,output_file,context_len_limit=100,que_len_limit=30
                                    ,char_len_limit=4,ans_len_limit = 2):
    writer = tf.python_io.TFRecordWriter(output_file)
    counter = 0
    with open(input_file, "r", errors='ignore', encoding='utf8') as file:
        for line in tqdm(file):

            try:
                line = json.loads(line)
            except:
                print("ERROR {0}\n{1}".format(counter,line))
                continue
            counter += 1
            if counter % 10000 == 0:
                print("current is {0}".format(counter))
            context_tokens = " ".join(jieba.cut(line['passage'])).split()
            # context_tokens = clean_detected_words(context_tokens)
            context_tokens_ids = []
            keys = word2id.keys()

            context_chars_ids = []

            if len(context_tokens) > context_len_limit:
                context_tokens = context_tokens[:int(context_len_limit/2)] + context_tokens[-int(context_len_limit/2):]

            for i in range(0, min(len(context_tokens), context_len_limit)):

                # 制作id向量
                if context_tokens[i] in keys and i:
                    context_tokens_ids.append(word2id[context_tokens[i]])
                else:
                    context_tokens_ids.append(word2id['unk'])

                c_ids = []
                for l in range(0, min(len(context_tokens[i]), char_len_limit)):
                    w = context_tokens[i][l]
                    if w in keys:
                        c_ids.append(word2id[w])
                    else:
                        c_ids.append(word2id['unk'])
                if len(context_tokens[i]) < char_len_limit:
                    for k in range(0, char_len_limit - len(context_tokens[i])):
                        c_ids.append(word2id['unk'])
                context_chars_ids.append(c_ids)

            if len(context_tokens) < context_len_limit:
                for k in range(0, context_len_limit - len(context_tokens)):
                    context_tokens_ids.append(word2id['unk'])
                    context_chars_ids.append([word2id['unk']] * char_len_limit)

            question_tokens = " ".join(jieba.cut(line['query'])).split()
            # question_tokens = clean_detected_words(question_tokens)
            question_tokens_ids = []
            question_char_ids = []
            for i in range(0, min(len(question_tokens), que_len_limit)):

                # 制作id向量
                if question_tokens[i] in keys:
                    question_tokens_ids.append(word2id[question_tokens[i]])
                else:
                    question_tokens_ids.append(word2id['unk'])

                c_ids = []
                for l in range(0, min(len(question_tokens[i]), char_len_limit)):

                    if question_tokens[i][l] in keys:
                        c_ids.append(word2id[question_tokens[i][l]])
                    else:
                        c_ids.append(word2id['unk'])
                if len(question_tokens[i]) < char_len_limit:
                    for k in range(0, char_len_limit - len(question_tokens[i])):
                        c_ids.append(word2id['unk'])
                question_char_ids.append(c_ids)

            if len(question_tokens) < que_len_limit:
                for k in range(0, que_len_limit - len(question_tokens)):
                    question_tokens_ids.append(word2id['unk'])
                    question_char_ids.append([word2id['unk']] * char_len_limit)

            alternaties = line['alternatives'].split("|")
            alter_ids = []

            for i in range(len(alternaties)):
                cut = " ".join(jieba.cut(alternaties[i].strip())).split(" ")
                line_ans = []

                for l in range(min(len(cut),ans_len_limit)):
                    if cut[l] in keys:
                        line_ans.append(word2id[cut[l]])
                    else:
                        line_ans.append(word2id['unk'])

                if len(line_ans)<ans_len_limit:
                    for l in range(ans_len_limit-len(cut)):
                        line_ans.append(word2id['unk'])
                alter_ids.append(line_ans)

            if len(alternaties)<3:
                for i in range(3-len(alternaties)):
                    alter_ids.append([0, 0])

            q_id = []
            q_id.append(int(line['query_id']))

            data = tf.train.Example(features=tf.train.Features(
                feature={
                    'context_tokens_ids': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[np.array(context_tokens_ids).tostring()])),
                    'context_chars_ids': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[np.array(context_chars_ids).tostring()])),
                    'ques_tokens_ids': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[np.array(question_tokens_ids).tostring()])),
                    'ques_chars_ids': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[np.array(question_char_ids).tostring()])),
                    'ans': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[np.array(alter_ids).tostring()])),
                    'q_id':tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[np.array(q_id).tostring()])),
                }))
            writer.write(data.SerializeToString())
        writer.close()
        print("TFRecords "+input_file+" has {} datas".format(counter))
        return counter

def ANSEstablishTFRecordsFromPICKLE_VER60(word2id,input_file,output_file,context_len_limit=100,que_len_limit=30,char_len_limit=4):
    writer = tf.python_io.TFRecordWriter(output_file)
    file = open(input_file, 'rb')
    datas = pickle.load(file)
    counter = 0

    for line in datas:
        if line[9] == -1:
            continue
        counter += 1
        if counter % 10000 == 0:
            print("current is {0}".format(counter))

        context_tokens = line[3]
        context_tokens_ids = []
        keys = word2id.keys()

        context_chars_ids = []

        for i in range(0,min(len(context_tokens),context_len_limit)):
            #制作id向量
            if context_tokens[i] in keys and i:
                context_tokens_ids.append(word2id[context_tokens[i]])
            else:
                context_tokens_ids.append(word2id['unk'])

            c_ids = []
            for l in range(0,min(len(context_tokens[i]),char_len_limit)):
                w = context_tokens[i][l]
                if w in keys:
                    c_ids.append(word2id[w])
                else:
                    c_ids.append(word2id['unk'])
            if len(context_tokens[i]) < char_len_limit:
                for k in range(0,char_len_limit-len(context_tokens[i])):
                    c_ids.append(word2id['unk'])
            context_chars_ids.append(c_ids)

        if len(context_tokens) < context_len_limit:
            for k in range(0,context_len_limit-len(context_tokens)):
                context_tokens_ids.append(word2id['unk'])
                context_chars_ids.append([word2id['unk']]*char_len_limit)

        question_tokens = line[2]
        question_tokens_ids = []
        question_char_ids = []
        for i in range(0,min(len(question_tokens),que_len_limit)):

            # 制作id向量
            if question_tokens[i] in keys:
                question_tokens_ids.append(word2id[question_tokens[i]])
            else:
                question_tokens_ids.append(word2id['unk'])

            c_ids = []
            for l in range(0,min(len(question_tokens[i]),char_len_limit)):

                if question_tokens[i][l] in keys:
                    c_ids.append(word2id[question_tokens[i][l]])
                else:
                    c_ids.append(word2id['unk'])
            if len(question_tokens[i]) < char_len_limit:
                for k in range(0,char_len_limit-len(question_tokens[i])):
                    c_ids.append(word2id['unk'])
            question_char_ids.append(c_ids)

        if len(question_tokens) < que_len_limit:
            for k in range(0,que_len_limit-len(question_tokens)):
                question_tokens_ids.append(word2id['unk'])
                question_char_ids.append([word2id['unk']]*char_len_limit)

        ans = line[5]
        q_id = [0]
        q_id[0] = line[7]

        data = tf.train.Example(features=tf.train.Features(
            feature={
                'context_tokens_ids':tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(context_tokens_ids).tostring()])),
                'context_chars_ids': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(context_chars_ids).tostring()])),
                'ques_tokens_ids':tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(question_tokens_ids).tostring()])),
                'ques_chars_ids': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(question_char_ids).tostring()])),
                'ans':tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(ans).tostring()])),
                'q_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(q_id).tostring()]))
            }
        ))
        writer.write(data.SerializeToString())
    writer.close()
    file.close()
    print("TFRecords "+input_file+" has {} datas".format(counter))
    return counter

def EstablishTestTFRecordsFromPickle_VER60(word2id,input_file,output_file,context_len_limit=100,que_len_limit=30,char_len_limit=4):
    writer = tf.python_io.TFRecordWriter(output_file)
    file = open(input_file, 'rb')
    datas = pickle.load(file)
    counter = 0

    for line in datas:

        counter += 1
        if counter % 10000 == 0:
            print("current is {0}".format(counter))
        context_tokens = line[3]
        context_tokens_ids = []
        keys = word2id.keys()

        context_chars_ids = []

        for i in range(0,min(len(context_tokens),context_len_limit)):
            #制作id向量
            if context_tokens[i] in keys and i:
                context_tokens_ids.append(word2id[context_tokens[i]])
            else:
                context_tokens_ids.append(word2id['unk'])

            c_ids = []
            for l in range(0,min(len(context_tokens[i]),char_len_limit)):
                w = context_tokens[i][l]
                if w in keys:
                    c_ids.append(word2id[w])
                else:
                    c_ids.append(word2id['unk'])
            if len(context_tokens[i]) < char_len_limit:
                for k in range(0,char_len_limit-len(context_tokens[i])):
                    c_ids.append(word2id['unk'])
            context_chars_ids.append(c_ids)

        if len(context_tokens) < context_len_limit:
            for k in range(0,context_len_limit-len(context_tokens)):
                context_tokens_ids.append(word2id['unk'])
                context_chars_ids.append([word2id['unk']]*char_len_limit)

        question_tokens = line[2]
        question_tokens_ids = []
        question_char_ids = []
        for i in range(0,min(len(question_tokens),que_len_limit)):

            # 制作id向量
            if question_tokens[i] in keys:
                question_tokens_ids.append(word2id[question_tokens[i]])
            else:
                question_tokens_ids.append(word2id['unk'])

            c_ids = []
            for l in range(0,min(len(question_tokens[i]),char_len_limit)):

                if question_tokens[i][l] in keys:
                    c_ids.append(word2id[question_tokens[i][l]])
                else:
                    c_ids.append(word2id['unk'])
            if len(question_tokens[i]) < char_len_limit:
                for k in range(0,char_len_limit-len(question_tokens[i])):
                    c_ids.append(word2id['unk'])
            question_char_ids.append(c_ids)

        if len(question_tokens) < que_len_limit:
            for k in range(0,que_len_limit-len(question_tokens)):
                question_tokens_ids.append(word2id['unk'])
                question_char_ids.append([word2id['unk']]*char_len_limit)

        q_id = [0]
        q_id[0] = line[7]

        data = tf.train.Example(features=tf.train.Features(
            feature={
                'context_tokens_ids': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[np.array(context_tokens_ids).tostring()])),
                'context_chars_ids': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[np.array(context_chars_ids).tostring()])),
                'ques_tokens_ids': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[np.array(question_tokens_ids).tostring()])),
                'ques_chars_ids': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[np.array(question_char_ids).tostring()])),
                'q_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(q_id).tostring()]))
            }
        ))
        writer.write(data.SerializeToString())
    writer.close()
    file.close()
    print("TFRecords "+input_file+" has {} datas".format(counter))
    return counter


def prepro(version,type=0,input = "None",embedding_table=None):
    '''
    数据预处理函数
    :param version: 版本号
    :param type: 0-valid，1-test，2-train
    :param input:指定的输入路径
    :return: word_mat-embedding table,counter-预处理了多少条数据
    '''
    counter = 0
    word_mat = []
    if version == 20:
        if not embedding_table:
            word_mat, w2id = GetEmbeddingFromNoHeadTXT(
                "./datasets/word2vec/jwe_size300.txt", word_dim=300)
        else:
            w2id = embedding_table

        if type == 0:
            counter = ANSEstablishTFRecordsFromJSON_VER20(w2id,"./datasets/aic18/valid_mini.json",
                                                          "./data/prepro/valid_ver20.tfrecords")
        elif type == 1:
            in_path = "./datasets/aic18/test_mini.json"
            if input != "None":
                in_path = input
            counter = ANSEstablishTFRecordsFromJSON_VER20(w2id,in_path ,
                                                "./data/prepro/testa_ver20.tfrecords")
        elif type == 2:
            counter = ANSEstablishTFRecordsFromJSON_VER20(w2id, "./datasets/aic18/train_mini.json",
                                                          "./data/prepro/train_ver20.tfrecords")
        return word_mat,counter,w2id
    if version == 60 or version == 646:
        if not embedding_table:
            word_mat, w2id = GetEmbeddingFromNoHeadTXT(
                "./datasets/word2vec/jwe_size300.txt", word_dim=300)
        else:
            w2id = embedding_table
        if type == 0:
            run_preprocess("./datasets/aic18/valid_mini.json","./data/pickles/valid_ver60.pickle")
            counter = ANSEstablishTFRecordsFromPICKLE_VER60(w2id, "./data/pickles/valid_ver60.pickle",
                                                "./data/prepro/valid_ver60.tfrecords")
        elif type == 1:
            in_path = "./datasets/aic18/test_mini.json"
            if input != "None":
                in_path = input
            run_preprocess(in_path, "./data/pickles/testa_ver60.pickle")
            counter = EstablishTestTFRecordsFromPickle_VER60(w2id, "./data/pickles/testa_ver60.pickle",
                                                "./data/prepro/testa_ver60.tfrecords")
        elif type == 2:
            run_preprocess("./datasets/aic18/train_mini.json", "./data/pickles/train_ver60.pickle")
            counter = ANSEstablishTFRecordsFromPICKLE_VER60(w2id, "./data/pickles/train_ver60.pickle",
                                                          "./data/prepro/train_ver60.tfrecords")
        return word_mat,counter,w2id