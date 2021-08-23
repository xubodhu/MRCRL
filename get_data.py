import numpy as np
import os
import pickle
import random



def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122



def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag

def mt(word2id,relation2id,data):
    fixlen =70
    f = data
    test_sen = []
    test_ans = []
    for content in f:
        content = content.strip().split()
        en1 = content[2]
        en2 = content[3]
        if content[4] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[4]]
        label = [0 for i in range(len(relation2id))]
        y_id = relation
        label[y_id] = 1
        test_ans.append(label)

        sentence = content[5:-1]
        en1pos = 0
        en2pos = 0
        for i in range(len(sentence)):
            if sentence[i] == en1:
                en1pos = i
            if sentence[i] == en2:
                en2pos = i
        output = []
        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])
        for i in range(min(fixlen, len(sentence))):
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]
            output[i][0] = word
        test_sen.append(output)

    return test_sen,test_ans

def load_word2vec():
    print(' reading word embedding data...')
    vec = []
    word2id = {}
    f = open('vec.txt', encoding='utf-8')
    info = f.readline()
    print('word vec info:', info)
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)
        content = content[1:]
        content = [float(i) for i in content]
        vec.append(content)
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)

    dim = 50
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)

    print('reading relation to id')
    relation2id = {}
    f = open('relation2question.txt', 'r', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split('\t')
        relation2id[content[0]] = int(content[1])
    f.close()
    return vec,word2id,relation2id


def mt_seperate(x_train):
    train_word = []
    train_pos1 = []
    train_pos2 = []
    for x in x_train:
        word = []
        pos1 = []
        pos2 = []
        for tmp in x:
            word.append(tmp[0])
            pos1.append(tmp[1])
            pos2.append(tmp[2])
        train_word.append(word)
        train_pos1.append(pos1)
        train_pos2.append(pos2)

    train_word = np.array(train_word)
    train_pos1 = np.array(train_pos1)
    train_pos2 = np.array(train_pos2)

    return train_word,train_pos1,train_pos2

