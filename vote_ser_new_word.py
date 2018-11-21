import json
import os
import argparse
import numpy as np

import jieba
import itertools

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser('ensemble')
    parser.add_argument('--ensemble_size', type=int, default=5, help='5')
    parser.add_argument('--mode', type=str, default='ensemble', help='ensemble/predict')
    parser.add_argument('--input', type=str, default='valid', help='valid/test')
    parser.add_argument('--input_file_dev', type=str, default='QANet/datasets/aic18/valid_mini.json',
                        help='/data1/lcn/project/aichallenger/data/input/valid.json')
    parser.add_argument('--input_file_test', type=str, default='QANet/datasets/aic18/test_mini.json',
                        help='/data1/lcn/project/aichallenger/data/input/test.json')
    parser.add_argument('--predict_file', type=str, default='predict_final',
                        help='list of files that contain the preprocessed dev data')
    parser.add_argument('--root_file', type=str, default='/data1/lcn/project/aichallenger/py/ensemble/valid/', help='')
    parser.add_argument('--template_dev', type=str,
                        default='QANet/datasets/aic18/template_dev.json',
                        help='')
    parser.add_argument('--template_test', type=str,
                        default='QANet/datasets/aic18/template_test.json',
                        help='')
    parser.add_argument('--files', nargs='+', default=['',''],
                        help='list of files that contain the preprocessed train data')
    return parser.parse_args()


def get_answer_label(seg_query, ansTokens):
    shuffledAnsToken_index = []
    shuffledAnsToken = []
    query = ""
    for i in seg_query:
        query += i
    label = None
    ansTokens = [x.strip() for x in ansTokens]
    unkownMark = False
    unkownIdx = -1
    unkownChar = ['无法确定', '无法确认', '不确定', '不能确定', 'wfqd', '无法选择', '无法确实', '无法取代', '取法确定', '无法确', '无法㾡', '无法去顶', '无确定',
                  '无法去顶', '我放弃', '无法缺定', '无法无额定', '无法判断', '不清楚', '无人确定', "不知道"]

    for idx, token in enumerate(ansTokens):
        for ch in unkownChar:
            if token.find(ch) != -1:
                unkownMark = True
                unkownIdx = idx
                break
        if unkownMark:
            break
            #            print("%s %s %s : %d %s"%(ansTokens[0],ansTokens[1],ansTokens[2],unkownIdx,ansTokens[unkownIdx]))
    minFindStart = 999999
    minIdx = -1
    if unkownMark == False:
        pass
        # print("%s %s %s unkonwn mark error" % (ansTokens[0], ansTokens[1], ansTokens[2]))
    else:
        for idx, token in enumerate(ansTokens):
            if unkownIdx == idx:
                continue
            tmpFindStart = query.find(token)
            if tmpFindStart == -1:
                tmpFindStart = 999999

            if minFindStart > tmpFindStart:
                minIdx = idx
                minFindStart = tmpFindStart
        if not (minIdx < 0 or minIdx > 2 or unkownMark < 0 or unkownMark > 2):
            if minIdx == 0:
                label = [1, 0, 0]
            elif unkownIdx == 0:
                label = [0, 0, 1]
            else:
                label = [0, 1, 0]
        else:
            minIdx = -999
            pessimisticDic = {"不会", "不可以", "不是", "假的", "不要", "不靠谱", "不能", "没有", "不需要", "没出", "不给", "不用", "不可能", "不好",
                              "不同意",
                              "不对", "不算", "不行", "不快", "不能", "没用", "不合适", "不正常", "不好", "不可", "不正确", "不高", "不难", "不属于",
                              "不合适",
                              "不值钱", "不友好", "不幸运", "不应该", "不值"}
            for idx, token in enumerate(ansTokens):
                if idx == unkownIdx:
                    continue
                for opt in pessimisticDic:
                    if token.find(opt) != -1:
                        minIdx = 3 - idx - unkownIdx
            if minIdx != -999:
                if minIdx == 0:
                    label = [1, 0, 0]
                elif unkownIdx == 0:
                    label = [0, 0, 1]
                else:
                    label = [0, 1, 0]
            else:
                minIdx = -999
                for idx, token in enumerate(ansTokens):
                    if token.find("不确定") == -1 and token.find("不能确定") == -1 and (
                            token.find("不") != -1 or token.find("否") != -1 or token.find(
                        "没") != -1 or token.find("错") != -1):
                        minIdx = 3 - idx - unkownIdx
                if minIdx != -999:
                    if minIdx == 0:
                        label = [1, 0, 0]
                    elif unkownIdx == 0:
                        label = [0, 0, 1]
                    else:
                        label = [0, 1, 0]
                else:
                    print("after last process ,still failed")
    try:
        if label != None:
            if minIdx == 0:
                if unkownIdx == 1:
                    shuffledAnsToken_index = [0, 2, 1]
                elif unkownIdx == 2:
                    shuffledAnsToken_index = [0, 1, 2]
            elif minIdx == 1:
                if unkownIdx == 0:
                    shuffledAnsToken_index = [1, 2, 0]
                elif unkownIdx == 2:
                    shuffledAnsToken_index = [1, 0, 2]
            elif minIdx == 2:
                if unkownIdx == 0:
                    shuffledAnsToken_index = [2, 1, 0]
                elif unkownIdx == 1:
                    shuffledAnsToken_index = [2, 0, 1]
            shuffledAnsToken = [ansTokens[i] for i in shuffledAnsToken_index]
    except:
        shuffledAnsToken_index = []

    return label, ansTokens, shuffledAnsToken_index, shuffledAnsToken


def modify_index_save(input_file, savefile):
    # print(input_file,savefile)
    outf = open(savefile, 'w', encoding='utf-8')
    inf = open(input_file, 'r', encoding='utf-8')
    for line in tqdm(inf):
        line = json.loads(line)
        alternatives = line['alternatives'].split('|')
        ques_word = list(jieba.cut(line['query']))
        query_id = line['query_id']
        label, ans, index, shu_ans = get_answer_label(ques_word, alternatives)
        if len(shu_ans) == 0:
            shu_ans = ans
        if label is None:
            label = [1, 0, 0]
        dict = {'query_id': query_id, 'ans_label': label, 'shu_ans': shu_ans, 'index': index, }
        # print(json.dumps(dict, ensure_ascii=False), file=outf)
        outf.write(json.dumps(dict, ensure_ascii=False))
        outf.write('\n')
    outf.close()


def ensemble_5(files, template_file, ensemble_size=3):
    threshold = ensemble_size // 2 + 1

    total = 0
    total_right = 0
    total_right_random = 0
    total_wrong = 0
    temp_data = {}
    template_f = open(template_file, 'r', encoding='utf-8')
    template = {}
    for line_ in template_f:
        line_ = json.loads(line_)
        shu_ans = line_['shu_ans']
        true_ans = shu_ans[np.argmax(line_['ans_label'])]
        template[line_['query_id']] = {'shu_ans': shu_ans, 'true_ans': true_ans}

    for index, path in enumerate(files):
        file = open(path, 'r', encoding='utf-8')
        for line in file:
            line = json.loads(line)
            id = line['query_id']

            if 'predict' in line.keys():
                if len(line['predict']) == 0:
                    predict_index = 0
                else:
                    predict_index = np.argmax(line['predict'])
                predict_word = template[id]['shu_ans'][predict_index]
            else:
                predict_word = line['pred_answer']

            if index == 0 or (index != 0 and id not in temp_data.keys()):
                temp_data[id] = {}
                temp_data[id]['true_ans'] = template[id]['true_ans']

            if predict_word not in temp_data[id].keys():
                temp_data[id][predict_word] = 1
            else:
                temp_data[id][predict_word] += 1

    for id, pre in temp_data.items():
        # print('----')
        total += 1
        label = pre['true_ans']
        for key, value in pre.items():
            if key != 'true_ans':
                if value >= threshold:
                    flag = 0
                    pre = key
                    break
                else:
                    flag = 1
                    pre = 0
        if flag == 1:  # if not get the rigth result of vote,select the first answer of alternatives(in valid is always first)
            total_right_random += 1
        elif flag == 0 and pre == label:
            total_right += 1
        elif flag == 0 and pre != label:
            total_wrong += 1
    print('{}/{} instances,random {},and the acc is {},total_wrong is {},'.format
          (total_right, total, total_right_random, (total_right+total_right_random) / total, total_wrong, ))
    return (total_right+total_right_random) / total


def ensemble_predict_5(predict_file, template_file, files, ensemble_size=3):
    ouf = open(predict_file, 'w', encoding='utf-8')
    threshold = ensemble_size // 2 + 1

    total = 0
    temp_data = {}

    template_f = open(template_file, 'r', encoding='utf-8')
    template = {}
    for line_ in template_f:
        line_ = json.loads(line_)
        shu_ans = line_['shu_ans']
        template[line_['query_id']] = {'shu_ans': shu_ans}

    for index, path in enumerate(files):
        file = open(path, 'r', encoding='utf-8')
        for line in file:
            line = json.loads(line)
            id = line['query_id']
            if index == 0 or (index != 0 and id not in temp_data.keys()):
                temp_data[id] = {}
                temp_data[id]['shu_ans'] = template[id]['shu_ans']

            if 'predict' in line.keys():
                if len(line['predict']) == 0:
                    predict_index = 0
                else:
                    predict_index = np.argmax(line['predict'])
                try:
                    shu_ans = template[id]['shu_ans']
                    predict_word = shu_ans[predict_index]
                except:
                    predict_word = template[id]['shu_ans'][0]
            else:
                predict_word = line['pred_answer']

            if predict_word not in temp_data[id].keys():
                temp_data[id][predict_word] = 1
            else:
                temp_data[id][predict_word] += 1
    for id, pre in temp_data.items():
        total += 1
        for key, value in pre.items():
            if key != 'shu_ans':
                if value >= threshold:
                    flag = 0
                    predict = key
                    break
                else:
                    flag = 1

        if flag == 1:
            predict = pre['shu_ans'][0]

        ouf.write((str(id) + '\t' + predict + '\n'))
    ouf.close()


if __name__ == '__main__':
    args = parse_args()

    files_ = []
    files_test=['data/v81/results/test_v81.txt',
                'data/v84/results/test_v84.txt',
                'QANet/answers/ver20/sorted_testa_v20.txt',
                'QANet/answers/ver60/testa_v60.txt',
                'QANet/answers/ver646/testa_v646.txt']
    files_valid=['data/v81/results/valid_v81.txt',
                 'data/v84/results/valid_v84.txt',
                 'QANet/answers/ver20/sorted_valid_v20.txt',
                 'QANet/answers/ver60/valid_v60.txt',
                 'QANet/answers/ver646/valid_v646.txt'
                 ]
    if args.mode == 'ensemble':
        modify_index_save(args.input_file_dev, args.template_dev)
        ensemble_5(files_valid, args.template_dev, args.ensemble_size)
    else:
        modify_index_save(args.input_file_test, args.template_test)
        ensemble_predict_5(args.predict_file, args.template_test, files_test, args.ensemble_size)