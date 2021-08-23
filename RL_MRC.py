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
from Test_MRC import test_MRC
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore", category=FutureWarning)

def get_mrc_na_data():
    na_data = []
    with open('train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if 'NA' in line:
            na_data.append(line)
    random.shuffle(na_data)
    return na_data

class MRC_dense(nn.Module):
    def __init__(self):
        super(MRC_dense,self).__init__()
        self.qa = nn.Linear(768,2)
        nn.init.xavier_normal_(self.qa.weight)
        
    def forward(self, sequence_out):
        logits = self.qa(sequence_out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits,end_logits

class RL_MRC(object):
    
    def load_source_dense(self):
        roberta_model = RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2', return_dict=True)
        model_states = roberta_model.qa_outputs.state_dict()
        print('Loading QA dense state dict')
        self.model.qa.load_state_dict(model_states,strict=True)
    
    def __init__(self,percentage):
        self.model = MRC_dense()
        self.tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        self.softmax = nn.Softmax(dim=1)
        self.cuda = torch.cuda.is_available()
        
        self.batch_size = 2560
        self.na_data = [] 
        self.val_data = []
        self.percentage = percentage
        
        self.un_check_string = [] 
        self.un_check_data = [] 
        self.step = 400
        self.T = 8 
        self.epoch = 1000 
        self.epoch2 = 600 
        self.rl_lr = 0.003
        self.id2data = None

    def train(self):
        na_data = get_mrc_na_data()
        self.load_source_dense()
        
        context_mask, ans_start_mask, ans_end_mask, sentence_idx, all_sequences, id2data = self.get_nyt_data()
        
        data_nums = all_sequences.shape[0]
        with open('valid_0.1.txt', 'r') as f:
            val_data = f.readlines()
            f.close()
        self.val_data = val_data
        self.id2data = id2data
        
        
        print('need reduce noise data size:{}'.format(data_nums))
        
        
        
        self.get_na_data()
        self.na_data = self.na_data[:64]
        print(' na data size is:{}'.format(len(self.na_data)))
        if self.cuda:
            print('GPU training...')
            self.model = self.model.cuda()
        
        optimizer = torch.optim.Adam(self.model.parameters(),lr = self.rl_lr)
        f1_list = []
        last_f1 = 0.0
        reward = 0.0
        torch.save(self.model.state_dict(),'rl_dense/step_'+str(0)+'.pth')
        
        for i in range(1,self.step+1):
            print('step:{} batch size:{}'.format(i,self.batch_size))
            optimizer.zero_grad()
            all_list = list(range(all_sequences.shape[0]))
            batch_idx = random.sample(all_list, self.batch_size)
            batch_sequences_embedding = all_sequences[batch_idx]
            batch_context_mask = context_mask[batch_idx]
            batch_start_mask = ans_start_mask[batch_idx]
            batch_end_mask = ans_end_mask[batch_idx]
            batch_sentences_ids = sentence_idx[batch_idx]
            if self.cuda:
                batch_sequences_embedding,batch_context_mask,batch_start_mask,batch_end_mask = \
                    batch_sequences_embedding.cuda(),batch_context_mask.cuda(),batch_start_mask.cuda(),batch_end_mask.cuda()
            
            start_logits , end_logits = self.model(batch_sequences_embedding)

            
            new_start_logits = start_logits.masked_fill(mask=batch_context_mask,value=torch.tensor(-1e9))
            new_end_logits = end_logits.masked_fill(mask=batch_context_mask,value=torch.tensor(-1e9))

            
            pro_start = self.softmax(new_start_logits)
            pro_end = self.softmax(new_end_logits)
            
            ans_start_pro = torch.mul(pro_start,batch_start_mask).sum(dim=1)
            ans_end_pro = torch.mul(pro_end,batch_end_mask).sum(dim=1)

            
            batch_scores = torch.mul(ans_start_pro,ans_end_pro)
            
            
            action_holder = self.select_by_pros(batch_scores)
            train_data = self.get_train_data(action_holder, batch_sentences_ids)
            action_holder = torch.from_numpy(np.array(action_holder)).float()
            if self.cuda:
                action_holder = action_holder.cuda()

            mx_f1,mx_epoch = train_union(train_data,self.val_data,'model/batch_cnn_model.ckpt')
            if i < self.T:
                reward = 0.0
                f1_list.append(mx_f1)
            elif i==self.T:
                reward = 0.0
                f1_list.append(mx_f1)
                last_f1 = sum(f1_list)/self.T
            else:
                reward = mx_f1 - last_f1
                last_f1 = (1 - 1 / self.T) * last_f1 + 1 / self.T * mx_f1

            with open('score.txt','a') as fl:
                fl.write('score:{}\n'.format(batch_scores))
                fl.close()
            print(batch_scores)
            print('score shape:{}'.format(batch_scores.shape))
            print('F1:{}  Reward :{}'.format(mx_f1,reward))
            reward_holder = torch.from_numpy(np.array([reward])).float()
            if self.cuda:
                reward_holder = reward_holder.cuda()
            pi = action_holder * batch_scores + (1.0 - action_holder) * (1.0 - batch_scores)
            loss = -1.0*torch.sum(torch.log(pi) * reward_holder)
            print('loss:{}'.format(loss))
            loss.backward()
            optimizer.step()
            with open('res.txt','a') as f:
                f.write('step:{}    batch size:{}'.format(i,self.batch_size)+'\n')
                f.write('F1:{}  Reward :{} mx_epoch:{}'.format(mx_f1,reward,mx_epoch)+'\n')
                f.write('loss:{}'.format(loss)+'\n\n')
                f.close()
            if i%50==0:
                
                torch.save(self.model.state_dict(),'rl_dense/step_'+str(i)+'.pth')
        return na_data
        

    def test_model_performance(self,mrc_na_data):
        print('start test...')
        model_dicts = self.model.state_dict()
        pers =[0.9,0.8,0.7,0.6,0.5]
        ans = test_MRC(model_dicts,pers,mrc_na_data)
        for res in ans:
            test_acc,test_p_r,per,data_size = res
            with open('tp_test_reduce_data_f1.txt','a') as f:
                f.write('test max ACC:{}  p_r:{}'.format(test_acc,test_p_r)+'\n')
                f.write('train data:{}  percentage:{}\n\n'.format(data_size,per))
                f.close()
        pass

    def get_train_data(self, select_idx, sentences_ids):
        
        output = self.na_data[:]
        for i in range(len(select_idx)):
            if select_idx[i] == 1:
                idx = sentences_ids[i].item()
                output.append(self.id2data[idx])
        print('train data size:{}'.format(len(output)))
        return output

    def select_by_rank(self, all_score):
        sorted_all_scores, indices = torch.sort(all_score, dim=0, descending=True)
        n = sorted_all_scores.shape[0]
        spilt_idx = math.floor(n * self.percentage)
        split_value = sorted_all_scores[spilt_idx]
        select_idx = []
        for i in range(n):
            if all_score[i] >= split_value:
                select_idx.append(1)
            else:
                select_idx.append(0)
        return select_idx
    
    def select_by_pros(self,all_score):
        n = all_score.shape[0]
        select_idx = []
        for i in range(n):
            key = random.random()
            if key < all_score[i].item():
                select_idx.append(1)
            else:
                select_idx.append(0)
        return select_idx

    def get_nyt_data(self):
        all_sequences = torch.load('embedding_data/train_sequences.pt')
        context_mask = torch.load('embedding_data/train_context_mask.pt')
        ans_start_mask = torch.load('embedding_data/train_ans_start_mask.pt')
        ans_end_mask = torch.load('embedding_data/train_ans_end_mask.pt')
        sentence_idx = torch.load('embedding_data/train_sentence_idx.pt')
        id2data = np.load('embedding_data/train_id2data.npy', allow_pickle=True).item()
        return context_mask,ans_start_mask,ans_end_mask,sentence_idx,all_sequences,id2data

    def get_na_data(self):
        self.na_data = []
        with open('train_0.9.txt','r',encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            if 'NA' in line:
                self.na_data.append(line)
        random.shuffle(self.na_data)



if __name__ == '__main__':
    
    st_time = time.time()
    print('time:',time.time()-st_time)
    print('MRC reduce noise...')
    percentage = 0.5
    mrc =RL_MRC(percentage)
    na_data = mrc.train()
    
    print('time:',time.time()-st_time)




