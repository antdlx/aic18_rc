import tensorflow as tf

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''
def get_record_parser20():
    def parse_example(serial_example):
        features = tf.parse_single_example(serial_example,features={
            'context_tokens_ids':tf.FixedLenFeature([],tf.string),
            'context_chars_ids':tf.FixedLenFeature([],tf.string),
            'ques_tokens_ids':tf.FixedLenFeature([],tf.string),
            'ques_chars_ids':tf.FixedLenFeature([],tf.string),
            'ans':tf.FixedLenFeature([],tf.string),
            'q_id': tf.FixedLenFeature([], tf.string)
        })
        context_tokens = tf.reshape(tf.decode_raw(features['context_tokens_ids'],tf.int64),[100])
        context_chars = tf.reshape(tf.decode_raw(features['context_chars_ids'],tf.int64),[100,4])
        ques_tokens = tf.reshape(tf.decode_raw(features['ques_tokens_ids'], tf.int64), [30])
        ques_chars = tf.reshape(tf.decode_raw(features['ques_chars_ids'], tf.int64), [30, 4])
        ans = tf.reshape(tf.decode_raw(features['ans'],tf.int64),[3,2])
        q_id = tf.reshape(tf.decode_raw(features['q_id'], tf.int64), [])
        return context_tokens,context_chars,ques_tokens,ques_chars,ans,q_id
    return parse_example

def get_record_parser60():
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
    return parse_example