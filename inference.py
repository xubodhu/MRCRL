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
from TFCNN_nyt import train_union
from TFCNN_nyt import train
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore", category=FutureWarning)
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
        self.rand_data = []
        pass
    def predict(self,na_data,x):
        
        self.na_data = na_data[:]
        (context_mask,ans_start_mask,ans_end_mask,sentence_idx,all_sequences,id2data) = x

        
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
        self.append_rand_sentence(select_idx,all_sentence_ids)
        print('output data size:{}'.format(len(self.output)))
        return self.output,self.rand_data

    def append_sentence(self,select_idx,sentences_ids):
        
        
        
        self.output = self.na_data[:]
        
        for i in range(len(select_idx)):
            if select_idx[i]==1:
                idx = sentences_ids[i].item()
                self.output.append(self.id2data[idx])
        print('train_data size:{}'.format(len(self.output)))
        pass
    def append_rand_sentence(self,select_idx,sentences_ids):
        self.rand_data = self.na_data[:]
        for i in range(len(select_idx)):
            key = random.random()
            if key < self.percentage:
                idx = sentences_ids[i].item()
                self.rand_data.append(self.id2data[idx])
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
    '''
    def select_by_pros(self,all_score):
        n = all_score.shape[0]
        select_idx = []
        for i in range(n):
            if 0.5 < all_score[i].item():
                select_idx.append(1)
            else:
                select_idx.append(0)
        return select_idx
    '''

def get_reduce_data():
    all_sequences = torch.load('embedding_data/test_sequences.pt')
    context_mask = torch.load('embedding_data/test_context_mask.pt')
    ans_start_mask = torch.load('embedding_data/test_ans_start_mask.pt')
    ans_end_mask = torch.load('embedding_data/test_ans_end_mask.pt')
    sentence_idx = torch.load('embedding_data/test_sentence_idx.pt')
    id2data = np.load('embedding_data/test_id2data.npy', allow_pickle=True).item()
    return (context_mask,ans_start_mask,ans_end_mask,sentence_idx,all_sequences,id2data)

def test_MRC(model_dicts,pers,na_data,x):
    with open('par.txt','a') as file:
        file.write('par is:{}\n\n'.format(model_dicts))
        file.close()
    ans = []
    for per in pers:
        
        mrc = MRC(percentage=per)
        mrc.model.load_state_dict(model_dicts,strict=True)
        tmp_data = na_data[:math.floor(len(na_data)*per)] 
        train_data,rand_data = mrc.predict(tmp_data,x)
        t_max,t_max_p_r= train(train_data,'model/new_data_cnn_model.ckpt')
        t_max2, t_max_p_r2 = train(rand_data, 'model/new_data_cnn_model.ckpt')
        ans.append((t_max,t_max_p_r,per,len(tmp_data),len(train_data)-len(tmp_data),
                    t_max2, t_max_p_r2,per,len(tmp_data),len(rand_data)-len(tmp_data)))
    return ans

def get_mrc_na_data():
    na_data = []
    with open('train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if 'NA' in line:
            na_data.append(line)
    random.shuffle(na_data)
    return na_data

def test_model_performance(model_dicts,mrc_na_data,x):
    print('start test...')
    pers =[0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
    ans = test_MRC(model_dicts,pers,mrc_na_data,x)
    for res in ans:
        test_acc,test_p_r,per,na_data_size,reduce_data_size,rand_acc,rand_pr,per2,na_size,reduce_size = res
        with open('cmp_rl_mrc.txt','a') as f:
            f.write('test max ACC:{} rand max ACC:{} \ntest pr:{} \nrand pr:{}\n'.format(test_acc,rand_acc,test_p_r,rand_pr))
            f.write('NA data:{} reduce data:{} percentage:{}\n\n'.format(na_data_size,reduce_data_size,per))
            f.write('rand NA data:{} rand reduce data:{} percentage:{}\n\n'.format(na_size, reduce_size, per2))
            f.close()
    pass
steps = [0,50,100,150,200,250,300,350,400]
na_data = []
x = get_reduce_data()
for step in steps:
    model_dicts = torch.load('rl_dense/step_'+str(step)+'.pth')
    test_model_performance(model_dicts,na_data,x)