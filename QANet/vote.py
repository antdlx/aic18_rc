import json

import jieba
from tqdm import tqdm

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
    minFindStart = 999999
    minIdx = -1
    if unkownMark == False:
        pass
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


def modify_index_save(templetfile, savefile):
    outf = open(savefile, 'w', encoding='utf-8')
    inf = open(templetfile, 'r', encoding='utf-8')
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
        print(json.dumps(dict, ensure_ascii=False), file=outf)


def modify(filename, templet, savefile):
    file_post = open(filename, 'r', encoding='utf-8')
    temp = open(templet, 'r', encoding='utf-8')
    savefile = open(savefile, 'w', encoding='utf-8')

    temp_data = {}
    for line_t in temp:
        line_t = json.loads(line_t)
        id = line_t['query_id']
        temp_data[id] = {'ans_label': line_t['ans_label'], 'shu_ans': line_t['shu_ans'], 'index': line_t['index']}

    for line in file_post:
        line = json.loads(line)
        id_f = line['query_id']
        result = temp_data[id_f]
        predict = [line['predict'][i] for i in result['index']]
        dict = {'query_id': id_f, 'predict': predict, 'ans_label': result['ans_label'], 'shu_ans': result['shu_ans']}
        print(json.dumps(dict, ensure_ascii=False), file=savefile)




