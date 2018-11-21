# -*- coding:utf8 -*-


# import sys
# sys.path.append('..')

import os

from capsuleNet.post_process import prepro

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from capsuleNet.dataset import BRCDataset
from capsuleNet.vocab import Vocab
from capsuleNet.rc_model84 import RCModel


def parse_args():
    """
    解析命令行变量

    """
    parser = argparse.ArgumentParser('Reading Comprehension on aic dataset')
    parser.add_argument('--mode', default="test",
                        help="Running mode test/dev/train/prepro")
    parser.add_argument('--input', default="../data/v84/testset/test_mini.json",
                        help='input path')
    parser.add_argument('--prepare', action='store_true', default=False,
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='3',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=10,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=4,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--load_epoch', default=1)
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=30,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=10,
                                help='max length of answer')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/trainset/train_pre.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/devset/dev_pre.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['../data/v84/testset/test_pre.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--vocab_dir', default='../data/v84/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/v84/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/v84/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/v84/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path', default='../data/v84/logging2',
                               help='path of the log file. If not set, logs are printed to console')

    return parser.parse_args()


def prepare(args):
    """
    检查数据，创建目录，准备词汇表和词嵌入
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("brc")
    logger.info('checking data file...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} is not exits.'.format(data_path)
    logger.info('establish folder...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('establish vocab...')
    brc_data = BRCDataset(args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files, args.test_files)
    vocab = Vocab(lower=True)
    print(vocab.size())
    for word in brc_data.word_iter('train'):
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)

    print(vocab.size())
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('filte {} words, final num is {}'.format(filtered_num,
                                                vocab.size()))
    logger.info('use w2v...')
    # vocab.randomly_init_embeddings(args.embed_size)
    vocab.load_pretrained_embeddings('../data/w2v/word2vec.model')

    print(vocab.size())
    logger.info('save word table...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('finish prepro!')


def train(args):
    """
    训练阅读理解模型
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("brc")

    file_handler = logging.FileHandler(args.log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(args)

    logger.info('loading datasets and vocab.data...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files)
    logger.info('changing to id...')
    brc_data.convert_to_ids(vocab)
    logger.info('init model...')
    rc_model = RCModel(vocab, args)
    logger.info('training model...')
    rc_model.train(brc_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                   save_prefix=args.algo,
                   dropout_keep_prob=args.dropout_keep_prob)
    logger.info('finish training!')


def evaluate(args):
    """
    对训练好的模型进行验证
    """
    logger = logging.getLogger("brc")
    logger.info('loading datasets and vocab.data...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'can not find valid file.'
    brc_data = BRCDataset(args.max_p_len, args.max_q_len, dev_files=args.dev_files)
    logger.info('change txt to id list')
    brc_data.convert_to_ids(vocab)
    logger.info('reloading model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo + '_{}'.format(args.epochs))
    logger.info('valid model...')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    dev_loss = rc_model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='valid_v84')
    logger.info('dev loss is: {}'.format(dev_loss))
    logger.info('save predict ans to {}'.format(os.path.join(args.result_dir)))


def predict(args):
    """
    预测测试文件的答案
    """
    logger = logging.getLogger("brc")
    logger.info('loading datasets and vocab.data...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'can not find test file.'
    brc_data = BRCDataset(args.max_p_len, args.max_q_len,
                          test_files=args.test_files)
    logger.info('change txt to id...')
    brc_data.convert_to_ids(vocab)
    logger.info('reloading model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo + '_{}'.format(args.epochs))
    logger.info('predict ans...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='test_v84')


def run():
    """
    预训练并运行整个系统.
    """
    args = parse_args()

    if args.mode == "dev":
        prepro(args.input,1,'../data/devset/dev_pre.json')
        args.evaluate = True
    elif args.mode == "test":
        prepro(args.input, 2, '../data/v84/testset/test_pre.json')
        args.predict = True
    elif args.mode == "prepro":
        args.prepare = True
        prepro(args.input, 3, '../data/v84/testset/test_pre.json')
    elif args.mode == "train":
        prepro(args.input, 1, '../data/trainset/train_pre.json')
        args.train = True

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)

if __name__ == '__main__':
    run()
