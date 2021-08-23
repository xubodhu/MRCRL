import json
import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.layers as layers
from tqdm import tqdm
import time
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import sys
from transformers import AutoModelForQuestionAnswering, AutoTokenizer,RobertaForQuestionAnswering
import warnings
import math
import random
import json
import re
import os
from TFCNN_nyt import train
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore", category=FutureWarning)

def get_relation2question():
    relation2id = {}
    id2question = {}
    with open('relation2question.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            line=line.strip()
            cols = line.split('\t')
            relation2id[cols[0]]= int(cols[1])
            if cols[0] == 'NA':
                continue
            question = re.sub(r'#.*#','{}',cols[2])
            entity = re.findall('#e(.+)#',cols[2])
            if len(entity)!=1:
                print('relation error!')
                exit()
            entity_id = int(entity[0])
            id2question[int(cols[1])] = (question,entity_id)
    for k,v in relation2id.items():
        if v == 0 :
            continue
        print(k,v,id2question[v])
    return relation2id,id2question

class MRC_dense(nn.Module):
    def __init__(self):
        super(MRC_dense,self).__init__()
        self.qa = nn.Linear(768,2)
    def forward(self, sequence_out):
        logits = self.qa(sequence_out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits,end_logits

class MRC(object):
    def __init__(self,percentage):

        self.tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        self.model = MRC_dense()
        self.softmax = nn.Softmax(dim=1)
        self.cuda = torch.cuda.is_available()
        self.batch_size = 5000
        self.output = []
        self.percentage = percentage
        self.id2data = None
        self.na_data = []
        pass
    def predict(self,na_data):

        self.na_data = na_data[:math.floor(len(na_data)*self.percentage)]
        context_mask,ans_start_mask,ans_end_mask,sentence_idx,all_sequences,id2data = \
            self.get_reduce_data()


        data_nums = all_sequences.shape[0]
        self.id2data = id2data

        if self.cuda:
            print('GPU training...')
            self.model = self.model.cuda()

        for i, j in self.model.named_parameters():
            j.requires_grad = False
        all_score = None
        all_sentence_ids =None
        for i in range(0,data_nums,self.batch_size):
            if i+self.batch_size >= data_nums:
                batch_context_mask = context_mask[i:]
                batch_start_mask = ans_start_mask[i:]
                batch_end_mask = ans_end_mask[i:]
                batch_sequences_embedding = all_sequences[i:]
                batch_sentence_ids = sentence_idx[i:]
            else:
                batch_context_mask = context_mask[i:i+self.batch_size]
                batch_start_mask = ans_start_mask[i:i+self.batch_size]
                batch_end_mask = ans_end_mask[i:i+self.batch_size]
                batch_sequences_embedding = all_sequences[i:i+self.batch_size]
                batch_sentence_ids = sentence_idx[i:i+self.batch_size]
            if self.cuda:
                batch_sequences_embedding,batch_context_mask,batch_start_mask,batch_end_mask = \
                    batch_sequences_embedding.cuda(),batch_context_mask.cuda(),batch_start_mask.cuda(),batch_end_mask.cuda()

            start_logits , end_logits = self.model(batch_sequences_embedding)
            if self.cuda:
                batch_sequences_embedding, batch_context_mask, batch_start_mask, batch_end_mask = \
                    batch_sequences_embedding.cpu(), batch_context_mask.cpu(), batch_start_mask.cpu(), batch_end_mask.cpu()
            if self.cuda:
                start_logits, end_logits = start_logits.cpu(), end_logits.cpu()
                torch.cuda.empty_cache()

            new_start_logits = start_logits.masked_fill(mask=batch_context_mask,value=torch.tensor(-1e9))
            new_end_logits = end_logits.masked_fill(mask=batch_context_mask,value=torch.tensor(-1e9))


            pro_start = self.softmax(new_start_logits)
            pro_end = self.softmax(new_end_logits)


            ans_start_pro = torch.mul(pro_start,batch_start_mask).sum(dim=1)
            ans_end_pro = torch.mul(pro_end,batch_end_mask).sum(dim=1)


            batch_scores = torch.mul(ans_start_pro,ans_end_pro)


            if all_score is None:
                all_score = batch_scores
                all_sentence_ids = batch_sentence_ids
            else:
                all_score = torch.cat((all_score,batch_scores),dim=0)
                all_sentence_ids = torch.cat((all_sentence_ids,batch_sentence_ids),dim=0)
                print('all score and ids shape:{}   {}'.format(all_score.shape,all_sentence_ids.shape))

        select_idx= self.select_by_rank(all_score)
        self.append_sentence(select_idx,all_sentence_ids)
        print('output data size:{}'.format(len(self.output)))
        return self.output

    def append_sentence(self,select_idx,sentences_ids):



        self.output = self.na_data[:]

        for i in range(len(select_idx)):
            if select_idx[i]==1:
                idx = sentences_ids[i].item()
                self.output.append(self.id2data[idx])
        print('train_data size:{}'.format(len(self.output)))
        pass

    def select_by_rank(self,all_score):
        sorted_all_scores,indices = torch.sort(all_score, dim=0, descending=True)
        n = sorted_all_scores.shape[0]
        spilt_idx = math.floor(n * self.percentage)
        split_value = sorted_all_scores[spilt_idx]
        select_idx = []
        for i in range(n):
            if all_score[i]>=split_value:
                select_idx.append(1)
            else:
                select_idx.append(0)
        return select_idx

    def get_reduce_data(self):
        all_sequences = torch.load('embedding_data/test_sequences.pt')
        context_mask = torch.load('embedding_data/test_context_mask.pt')
        ans_start_mask = torch.load('embedding_data/test_ans_start_mask.pt')
        ans_end_mask = torch.load('embedding_data/test_ans_end_mask.pt')
        sentence_idx = torch.load('embedding_data/test_sentence_idx.pt')
        id2data = np.load('embedding_data/test_id2data.npy', allow_pickle=True).item()
        return context_mask,ans_start_mask,ans_end_mask,sentence_idx,all_sequences,id2data


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
def initial_sentence(relation2id,word2id,path_name,types):

    fixlen = 70
    f = open(path_name, 'r', encoding='utf-8')
    train_sen = []
    train_ans = []
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        en1 = content[0]
        en2 = content[1]
        relation = relation2id[content[2]]
        label = [0 for i in range(len(relation2id))]
        y_id = relation
        label[y_id] = 1
        train_ans.append(label)

        sentence = content[3:-1]
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
        train_sen.append(output)
    train_word = []
    train_pos1 = []
    train_pos2 = []
    for tmp in train_sen:
        tmp_train_word = []
        tmp_train_pos1 = []
        tmp_train_pos2 = []
        for i in tmp:
            tmp_train_word.append(i[0])
            tmp_train_pos1.append(i[1])
            tmp_train_pos2.append(i[2])
        train_word.append(tmp_train_word)
        train_pos1.append(tmp_train_pos1)
        train_pos2.append(tmp_train_pos2)
    np.save("data/train_x.npy", train_sen)
    np.save("cnndata/cnn_"+types+"_word.npy",train_word)
    np.save("cnndata/cnn_"+types+"_pos1.npy",train_pos1)
    np.save("cnndata/cnn_"+types+"_pos2.npy",train_pos2)
    np.save("cnndata/cnn_"+types+"_y.npy", train_ans)

def test_MRC(model_dicts,pers,na_data):
    ans = []
    for per in pers:
        mrc = MRC(percentage=per)
        mrc.model.load_state_dict(model_dicts,strict=True)
        train_data = mrc.predict(na_data)
        t_max,t_max_p_r= train(train_data,'model/new_data_cnn_model.ckpt')
        ans.append((t_max,t_max_p_r,per,len(train_data)))
    return ans