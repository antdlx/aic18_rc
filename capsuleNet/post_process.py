import  json

import jieba
from tqdm import tqdm


def word_tokenize(sent):
    return list(jieba.cut(sent))


def process_file(filename, data_type, save_filename):
    print("Generating {} examples...".format(data_type))
    save_ = open(save_filename, 'w', encoding='utf-8')
    datas = []
    with open(filename, "r",encoding='utf8',errors='ignore') as fh:
        for line in tqdm(fh):
            try:
                dic = json.loads(line)
            except:
                continue

            question = dic['query']
            # passage = re.sub(pattern, '', dic['passage'])
            passage = dic['passage']
            alternatives = dic['alternatives']
            pos_alternatives = alternatives.split('|')
            segmented_alternatives = []
            for alt in pos_alternatives:
                segmented_alternatives.append(word_tokenize(alt))
            ques_word = word_tokenize(question)
            passage_word = word_tokenize(passage)
            if data_type == 'test':
                data = {"segmented_passage": passage_word, "segmented_query": ques_word,
                        "alternatives": alternatives, "pos_alternatives": pos_alternatives,
                        "segmented_alternatives": segmented_alternatives,"query_id":dic['query_id']}
            else:
                data = {"segmented_passage": passage_word, "segmented_query": ques_word,
                        "alternatives": alternatives, "pos_alternatives": pos_alternatives,
                        "segmented_alternatives": segmented_alternatives, "answer": dic['answer'],"query_id":dic['query_id']}
            save_.write(json.dumps(data,ensure_ascii=False)+"\n")
            datas.append(data)
        # random.shuffle(examples)
        print("{} data in total".format(len(datas)))
    save_.close()
    return datas



def prepro(ipath,mode,opath):
    # 1-valid 2-test 3-prepro
    if mode == 1:
        process_file(ipath, 'dev', opath)
    elif mode == 2:
        process_file(ipath, 'test', opath)
    elif mode == 3:
        process_file("../data/devset/dev_mini.json", 'dev', "../data/devset/dev_pre.json")
        process_file("../data/trainset/train_mini.json", 'train', "../data/trainset/train_pre.json")
        process_file(ipath, 'test', opath)
