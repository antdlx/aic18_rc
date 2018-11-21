import argparse
import os
import tensorflow as tf

from QANet.prepro import prepro

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

from QANet.main_debug_cla_v60 import valid, test, train

flags = tf.flags
parser = argparse.ArgumentParser('Reading Comprehension on aic dataset')
parser.add_argument('--mode', default="test",
                        help="Running mode test/valid/train/debug")
parser.add_argument('--input', default="./datasets/aic18/test_mini.json",
                        help='data input path')

answer_dir = "./answers/ver60"
model_dir = "./models/ver60"

train_dir = "models"
model_name = "ver60"
answer_file = os.path.join(answer_dir, "testa_v60.txt")
valid_file = os.path.join(answer_dir, "valid_v60.txt")
dir_name = os.path.join(train_dir, model_name)
log_dir = os.path.join(dir_name, "event")
save_dir = os.path.join(dir_name, "model")

if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)
if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)

flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")

flags.DEFINE_string("mode", "train", "Running mode test/valid")
flags.DEFINE_string("input", "./datasets/aic18/test.json", "Running mode test/valid")
flags.DEFINE_string("model_dir", model_dir, "Directory for saving model")
flags.DEFINE_string("valid_file", valid_file, "Out file for answer")
flags.DEFINE_string("answer_file", answer_file, "Out file for answer")

flags.DEFINE_integer("glove_char_size", 94, "Corpus size for Glove")
flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 300, "Embedding dimension for char")

flags.DEFINE_integer("para_limit",100, "200Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 30, "Limit length for question")
flags.DEFINE_integer("ans_limit", 3, "Limit length for answers")
flags.DEFINE_integer("test_para_limit", 1000, "Limit length for paragraph in test file")
flags.DEFINE_integer("test_ques_limit", 100, "Limit length for question in test file")
flags.DEFINE_integer("char_limit", 4, "Limit length for character")
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
flags.DEFINE_integer("char_count_limit", -1, "Min count for char")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")

flags.DEFINE_integer("batch_size", 256, "128Batch size")
flags.DEFINE_integer("epoch", 25, "epoch size")
flags.DEFINE_integer("period", 50, "100period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_float("dropout", 0.1, "Dropout prob across the layers")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("learning_decay_rate", 0.9, "Learning rate")
flags.DEFINE_float("end_learning_rate", 0.0001, "Learning rate")
flags.DEFINE_float("decay", 0.9999, "Exponential moving average decay")
flags.DEFINE_float("l2_norm", 3e-7, "L2 norm scale")
flags.DEFINE_integer("hidden", 96, "Hidden size")
flags.DEFINE_integer("num_heads", 1, "Number of heads in self attention")
flags.DEFINE_integer("early_stop", 8, "Checkpoints for early stop")

def main(_):
    config = flags.FLAGS
    args = parser.parse_args()
    if args.mode == "valid":
        word_mat,counter,_ = prepro(60)
        valid(config,word_mat,counter)
    elif args.mode == "test":
        word_mat,counter,_ = prepro(60,type=1,input=args.input)
        test(config,word_mat,counter)
    elif args.mode == "train":
        word_mat, train_counter, w2id = prepro(60, type=2)
        _, dev_counter, _ = prepro(60, type=0, embedding_table=w2id)
        train(config, word_mat, train_counter, dev_counter)
    elif args.mode == "debug":
        word_mat, train_counter, w2id = prepro(60, type=2)
        _, dev_counter, _ = prepro(60, type=0, embedding_table=w2id)
        config.batch_size = 2
        config.epoch = 3
        train(config, word_mat, train_counter, dev_counter)
    else:
        print("Unknown mode")
        exit(0)



if __name__ == "__main__":
    tf.app.run()
