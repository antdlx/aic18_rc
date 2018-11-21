import tensorflow as tf
import ujson as json
from tqdm import tqdm

from QANet.util import get_record_parser60
from QANet.model_cla_ver646 import Model

import os

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''
'''
train:249431
dev:29930
testa:10000249431
'''

def train(config,word_mat,train_counter,dev_counter):
    print("Building model...")
    parser = get_record_parser60()
    graph = tf.Graph()
    with graph.as_default() as g:
        #读取tfrecords文件
        train_dataset = tf.data.TFRecordDataset(
            "./data/prepro/train_ver60.tfrecords")
        train_dataset = train_dataset.shuffle(train_counter).map(parser).batch(config.batch_size)

        dev_dataset = tf.data.TFRecordDataset(
            "./data/prepro/valid_ver60.tfrecords")
        dev_dataset = dev_dataset.shuffle(dev_counter).map(parser).batch(config.batch_size)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_initializer = iterator.make_initializer(train_dataset)
        dev_initializer = iterator.make_initializer(dev_dataset)

        model = Model(config, iterator, word_mat, word_mat, graph = g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        patience = 0

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(config.log_dir)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            #训练多少轮就验证一下
            eval_per_epoch = 1
            #绘制验证集的tensorboard需要用的变量
            best_dev_acc = 0
            best_dev_loss = 100000

            #如果有先前训练好的模型，那就重载模型
            if os.path.exists(os.path.join(config.save_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            global_step = max(sess.run(model.global_step), 1)

            for e in range(0,config.epoch):
                sess.run(train_initializer)
                #训练
                num_steps = int(train_counter/config.batch_size)
                for _ in tqdm(range(num_steps)):
                    global_step = sess.run(model.global_step) + 1
                    loss, train_op,ansp,acc = sess.run([model.loss, model.train_op, model.ansp, model.acc], feed_dict={
                                              model.dropout: config.dropout})
                    if global_step % config.period == 0:
                        loss_sum = tf.Summary(value=[tf.Summary.Value(
                            tag="model/loss", simple_value=loss), ])
                        loss_acc = tf.Summary(value=[tf.Summary.Value(
                            tag="model/acc", simple_value=acc), ])
                        writer.add_summary(loss_sum, global_step)
                        writer.add_summary(loss_acc, global_step)
                    if global_step % 100 ==0 :
                        print("===TRAIN=== global_step is {0}, loss is {1}, ansp is {2}, acc is {3}"
                              .format(global_step, loss, ansp,acc))

                filename = os.path.join(
                        config.save_dir, "model_{}.ckpt".format(e))
                saver.save(sess, filename)

                #验证
                if e % eval_per_epoch == 0:
                    sess.run(dev_initializer)
                    times = int(dev_counter/config.batch_size)
                    all_dev_loss = 0
                    all_dev_acc = 0
                    for _ in range(times):
                        loss, ansp, acc = sess.run([model.loss, model.ansp, model.acc], feed_dict={
                            model.dropout: 0.0})
                        all_dev_acc += acc
                        all_dev_loss += loss

                    all_ave_dev_loss = all_dev_loss/times
                    all_ave_dev_acc = all_dev_acc/times
                    summary_all_dev_ave_loss = tf.Summary(value=[tf.Summary.Value(
                        tag="model/dev_ave_loss", simple_value=all_ave_dev_loss), ])
                    summary_all_dev_ave_acc = tf.Summary(value=[tf.Summary.Value(
                        tag="model/dev_ave_acc", simple_value=all_ave_dev_acc), ])
                    writer.add_summary(summary_all_dev_ave_loss,e)
                    writer.add_summary(summary_all_dev_ave_acc, e)
                    print("==DEV{0}== ave loss is {1}, ave acc is{2}".format(e,all_ave_dev_loss,all_ave_dev_acc))
                    if  all_ave_dev_loss > best_dev_loss and all_ave_dev_acc < best_dev_acc:
                        patience += 1
                        if patience > config.early_stop:
                            break
                    else:
                        patience = 0
                        best_dev_loss = min(best_dev_loss, all_ave_dev_loss)
                        best_dev_acc = max(best_dev_acc,all_ave_dev_acc)

def test(config,word_mat,counter):
    def parse_example(serial_example):
        features = tf.parse_single_example(serial_example, features={
            'context_tokens_ids': tf.FixedLenFeature([], tf.string),
            'context_chars_ids': tf.FixedLenFeature([], tf.string),
            'ques_tokens_ids': tf.FixedLenFeature([], tf.string),
            'ques_chars_ids': tf.FixedLenFeature([], tf.string),
            'q_id': tf.FixedLenFeature([], tf.string)
        })
        context_tokens = tf.reshape(tf.decode_raw(features['context_tokens_ids'], tf.int64), [100])
        context_chars = tf.reshape(tf.decode_raw(features['context_chars_ids'], tf.int64), [100, 4])
        ques_tokens = tf.reshape(tf.decode_raw(features['ques_tokens_ids'], tf.int64), [30])
        ques_chars = tf.reshape(tf.decode_raw(features['ques_chars_ids'], tf.int64), [30, 4])
        q_id = tf.reshape(tf.decode_raw(features['q_id'], tf.int64), [])
        return context_tokens, context_chars, ques_tokens, ques_chars, q_id


    graph = tf.Graph()
    batch_size = 1
    print("Loading model...")
    with graph.as_default() as g:
        train_dataset = tf.data.TFRecordDataset(
            "./data/prepro/testa_ver60.tfrecords")
        train_dataset = train_dataset.map(parse_example).batch(batch_size)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_initializer = iterator.make_initializer(train_dataset)

        model = Model(config, iterator, word_mat, word_mat,trainable=False, graph=g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        out_file = open(config.answer_file,"w",encoding="utf8",errors='ignore')
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(train_initializer)
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))

            results = []
            for step in tqdm(range(counter)):

                qa_id, logits = sess.run([model.id,model.logits])
                for i in range(batch_size):
                    result = {}
                    result['query_id'] = qa_id.tolist()[i]
                    result['predict'] = logits.tolist()[i]

                    results.append(result)
            for i in range(len(results)):
                r = json.dumps(results[i], ensure_ascii=False)
                out_file.write("{}\n".format(r))
            out_file.close()

def valid(config,word_mat,counter):
    def parse_example(serial_example):
        features = tf.parse_single_example(serial_example, features={
            'context_tokens_ids': tf.FixedLenFeature([], tf.string),
            'context_chars_ids': tf.FixedLenFeature([], tf.string),
            'ques_tokens_ids': tf.FixedLenFeature([], tf.string),
            'ques_chars_ids': tf.FixedLenFeature([], tf.string),
            'ans': tf.FixedLenFeature([], tf.string),
            'q_id': tf.FixedLenFeature([], tf.string)
        })
        context_tokens = tf.reshape(tf.decode_raw(features['context_tokens_ids'], tf.int64), [100])
        context_chars = tf.reshape(tf.decode_raw(features['context_chars_ids'], tf.int64), [100, 4])
        ques_tokens = tf.reshape(tf.decode_raw(features['ques_tokens_ids'], tf.int64), [30])
        ques_chars = tf.reshape(tf.decode_raw(features['ques_chars_ids'], tf.int64), [30, 4])
        ans = tf.reshape(tf.decode_raw(features['ans'], tf.int64), [3])
        q_id = tf.reshape(tf.decode_raw(features['q_id'], tf.int64), [])
        return context_tokens, context_chars, ques_tokens, ques_chars, ans,q_id


    graph = tf.Graph()
    batch_size = 1
    print("Loading model...")
    with graph.as_default() as g:
        train_dataset = tf.data.TFRecordDataset(
            "./data/prepro/valid_ver60.tfrecords")
        train_dataset = train_dataset.map(parse_example).batch(batch_size)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_initializer = iterator.make_initializer(train_dataset)

        model = Model(config, iterator, word_mat, word_mat,trainable=False,wrong=True, graph=g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        out_file = open(config.valid_file,"w",encoding="utf8",errors='ignore')
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(train_initializer)
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))

            results = []
            for step in tqdm(range(counter)):
                try:
                    qa_id, logits, label = sess.run([model.qa_id, model.logits, model.ans])
                except:
                    print(step)
                    exit(1)
                for i in range(batch_size):
                    result = {}
                    result['query_id'] = qa_id.tolist()[i]
                    result['predict'] = logits.tolist()[i]
                    result['label'] = label.tolist()[i]
                    results.append(result)
            for i in range(len(results)):
                r = json.dumps(results[i], ensure_ascii=False)
                out_file.write("{}\n".format(r))
            out_file.close()