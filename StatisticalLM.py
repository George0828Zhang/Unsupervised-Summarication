#!/usr/bin/env python
# coding: utf-8

import json
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

class StatisticalLM():
    def __init__(self, lm_record, vocab, lm_lambda = 0.4):
        self.lm_record = lm_record
        self.vocab = vocab
        self.lm_lambda = lm_lambda
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, word_ids):
        output = torch.zeros((word_ids.shape[0], len(self.vocab))).float()
        for b, sen in enumerate(word_ids):
            
            if len(sen) == 0:
                word_list = self.lm_record
            else:
                word_list = self.lm_record['next'][str(sen[-1].item())]
            
            for (k, v) in word_list['next'].items():
                output[b, int(k)] = v['count']
            output[b] /= word_list['count']
            
            for i in (output[b] == 0).nonzero():
                if str(i.item()) in self.lm_record['next']:
                    output[b, i] = self.lm_record['next'][str(i.item())]['count'] / self.lm_record['count'] * self.lm_lambda
        
        output = F.normalize(output, p=1, dim = -1)
        return output

def add_record(record, i):
    record['count'] += 1
    if len(i) > 0 and i[0] >= 0:
        if i[0] not in record['next']:
            record['next'][i[0]] = {'count': 0, 'next': {}}
        record['next'][i[0]] = add_record(record['next'][i[0]], i[1:])
    return record

if __name__ == '__main__':
    
    data_dir = '/hdd/unsupervised-summarization/data-giga/'

    with open(data_dir+'vocab.json') as f:
        vocab = json.load(f)
    with open(data_dir+'train_seq.json') as f:
        train = json.load(f)
        
    vocab_inv = {a:b for b,a in vocab.items()}

    lm_record = {'count':0, 'next': {}}

    for s in tqdm(train['document']):
        s.extend([-1]*1)
        for i in range(len(s)-1):
            lm_record = add_record(lm_record, s[i:i+2])

    with open(data_dir+'lm_record.json', 'w') as f:
        json.dump(lm_record, f)
        
    statistical_LM = StatisticalLM(lm_record, vocab)
    torch.save(statistical_LM, data_dir+'statistical_LM')