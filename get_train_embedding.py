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

def save_train_data(file_path,pattern ='origin'):
    origin_data = []
    nums = 0
    id2data = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if pattern == 'origin':
            for line in lines:
                line = line.strip()
                cols = line.split('\t')
                origin_data.append((cols[2], cols[3], cols[4], " ".join(cols[5:-1]), nums))
                id2data[nums] = line
                nums += 1
        elif pattern == 'new':
            for line in lines:
                line = line.strip()
                cols = line.split('\t')
                origin_data.append((cols[0], cols[1], cols[2], " ".join(cols[3:-1]), nums))
                id2data[nums] = line
                nums += 1
        f.close()

    print(origin_data[:5])
    print('origin data size:{}'.format(len(origin_data)))
    data = []
    na_output = []
    relation2id, id2question = get_relation2question()
    for sample in origin_data:
        e1, e2, rel, context, sen_id = sample
        if rel not in relation2id.keys():
            rel = 'NA'
        idx = relation2id[rel]
        if rel == 'NA':
            na_output.append(id2data[sen_id])
        else:
            answer = ''
            question, entity_id = id2question[idx]
            if entity_id == 1:
                question = question.format(e1)
                answer = e2
            elif entity_id == 2:
                question = question.format(e2)
                answer = e1
            else:
                exit()
            data.append((question, context, answer, sen_id))
    print('all data size except NA:{}'.format(len(data)))
    print('NA data size:{}'.format(len(na_output)))


    all_inputs = []
    att_mask = []
    ans_start_mask = []
    ans_end_mask = []
    context_mask = []
    sentence_idx = []
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

    error_num = 0
    for triple in data:
        question, context, answer, sentence_id = triple
        input_pair = list([(question, context)])
        encode = tokenizer.batch_encode_plus(input_pair, max_length=128, truncation=True,
                                                  pad_to_max_length=True)
        inputs, attention = encode['input_ids'], encode['attention_mask']
        inputs, attention = inputs[0], attention[0]
        inputs_len = len(inputs)
        if attention[-1] == 1:
            continue
        flag = 1
        token_type = []
        for i in range(inputs_len):
            if inputs[i] == 2 and inputs[i - 1] == 2:
                token_type.append(flag)
                flag = 0
            elif inputs[i] == 2:
                flag = 1
                token_type.append(flag)
            else:
                token_type.append(flag)

        ans1 = tokenizer(answer)['input_ids'][1:-1]
        ans2 = tokenizer('the ' + answer)['input_ids'][2:-1]
        start_idx = -1
        end_idx = -1

        for i in range(inputs_len):
            length = len(ans1)
            if inputs[i:i + length] == ans1:
                start_idx = i
                end_idx = i + length - 1

        if start_idx == -1 or end_idx == -1:
            for i in range(inputs_len):
                length = len(ans2)
                if inputs[i:i + length] == ans2:
                    start_idx = i
                    end_idx = i + length - 1
        if start_idx == -1 or end_idx == -1:
            print(context)
            print(type(context))
            print(triple)
            print(inputs)
            print(ans1)
            print(ans2)
            print('Find answer error !!!')
            error_num+=1
            continue
        ans_start = [0] * inputs_len
        ans_end = [0] * inputs_len
        ans_start[start_idx] = 1
        ans_end[end_idx] = 1

        all_inputs.append(inputs)
        att_mask.append(attention)
        ans_start_mask.append(ans_start)
        ans_end_mask.append(ans_end)
        context_mask.append(token_type)
        sentence_idx.append(sentence_id)
    print('error_num is:{}'.format(error_num))

    all_inputs = np.array(all_inputs)
    att_mask = np.array(att_mask)
    ans_start_mask = np.array(ans_start_mask)
    ans_end_mask = np.array(ans_end_mask)
    context_mask = np.array(context_mask)
    sentence_idx = np.array(sentence_idx)

    all_inputs = torch.from_numpy(all_inputs).long()
    att_mask = torch.from_numpy(att_mask).long()
    context_mask = torch.from_numpy(context_mask).bool()
    ans_start_mask = torch.from_numpy(ans_start_mask).float()
    ans_end_mask = torch.from_numpy(ans_end_mask).float()
    sentence_idx = torch.from_numpy(sentence_idx).long()
    print('Start get roberta embedding')
    print('need reduce data inputs shape:{}'.format(all_inputs.shape))
    roberta_model = RobertaForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2', return_dict=True)
    for i, j in roberta_model.named_parameters():
        j.requires_grad = False
    # get sequences embedding
    all_sequences = []
    for i in range(0,all_inputs.shape[0],1000):
        print('step:{}'.format(int(i/1000)))
        if i==0:
            all_sequences = roberta_model.roberta(all_inputs[i:i+1000], att_mask[i:i+1000]).last_hidden_state
        elif i+1000<=all_inputs.shape[0]:
            batch_sequences = roberta_model.roberta(all_inputs[i:i + 1000], att_mask[i:i + 1000]).last_hidden_state
            all_sequences = torch.cat((all_sequences,batch_sequences),dim=0)
        else:
            batch_sequences = roberta_model.roberta(all_inputs[i:], att_mask[i:]).last_hidden_state
            all_sequences = torch.cat((all_sequences, batch_sequences), dim=0)
    print('need reduce data out sequences shape:{}'.format(all_sequences.shape))
    print(all_sequences.shape)
    torch.save(all_sequences,'embedding_data/train_sequences.pt')
    torch.save(context_mask,'embedding_data/train_context_mask.pt')
    torch.save(ans_start_mask,'embedding_data/train_ans_start_mask.pt')
    torch.save(ans_end_mask,'embedding_data/train_ans_end_mask.pt')
    torch.save(sentence_idx,'embedding_data/train_sentence_idx.pt')
    np.save('embedding_data/train_id2data.npy',id2data)
    #return all_sequences,all_inputs, att_mask, context_mask, ans_start_mask, ans_end_mask, sentence_idx, na_output, id2data

save_train_data('train_0.9.txt')