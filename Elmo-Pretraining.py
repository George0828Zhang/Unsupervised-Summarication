#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
import torch
import torch.nn as nn
import json
import random
from preprocessors import BOS, EOS, PAD, UNK
import math
from tqdm.auto import tqdm, trange
from ELMo import LanguageModel
from dataset import *


# In[2]:


batch_size = 64
batch_size_inf = 64


# In[3]:


import wandb

wandb.init(entity="george0828zhang", project="contextual-matching-policy-gradient")
wandb.config.update({
    "batch_size": batch_size,
    })


# In[4]:


data_dir = "../data-giga/"
preload = "../preload_LM"
outdir = "traindELMo/"
vocab = json.load(open(data_dir+"vocab.json", "r"))
vocab_size = len(vocab)
training_set = PretrainDataset(data_dir+"train_seq.json", 50, 50, vocab[PAD]) #train_seq
validation_set = PretrainDataset(data_dir+"valid_seq.json", 50, 50, vocab[PAD])


# In[5]:


training_generator = Loader(training_set, batch_size=batch_size, shuffle=True)
validation_generator = Loader(validation_set, batch_size=batch_size_inf, shuffle=False)
total_train = int(math.ceil(training_set.size / batch_size))
total_valid = int(math.ceil(validation_set.size / batch_size_inf))


# In[6]:


device = torch.device("cuda")
model = torch.load(preload).to(device) #LanguageModel(vocab, unidir=True).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab[PAD]).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)


# In[7]:


wandb.watch([model])


# In[8]:


def validation():
    model.eval()
    total_loss = []
    with torch.no_grad():   
        trange = tqdm(validation_generator, total=total_valid)
        for src, tgt in trange:
            src = src.to(device)
            tgt = tgt.to(device)
            
            logits = model(src)
            loss = criterion(logits.view(-1, vocab_size), tgt.view(-1))            

            total_loss.append(loss.item())
            
    return np.mean(total_loss)


# In[9]:


vocab_inv = {a:b for b,a in vocab.items()}
def id2sent(ids):
    toks = (vocab_inv[i] for i in ids)
    return " ".join(toks)
def tstring(reward):
    return ", ".join([format(f, ".5g") for f in reward.cpu().numpy()])


# In[10]:


start = 1
epochs = 10


# In[11]:


# if start != 1:
#     smodel = torch.load("trainedELMo/Model"+str(start-1))
#     model.load_state_dict(smodel['model'])


# In[12]:


for e in range(start, epochs+1):
    model.train()
    loss_history = []
    trange = tqdm(training_generator, total=total_train, desc="epoch {}".format(e))
    
    for src, tgt in trange:
        src = src.to(device)
        tgt = tgt.to(device)
        
        logits = model(src)
        loss = criterion(logits.view(-1, vocab_size), tgt.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        loss_history.append(loss.item())
        trange.set_postfix(**{'loss':'{:.5f}'.format(loss.item())})
        
        _, ys = torch.max(logits, dim=-1)
        
        ### logging        
        wandb.log({
            "input":id2sent(src[0].cpu().numpy()),
            "output":id2sent(ys[0].cpu().numpy()),
            "target":id2sent(tgt[0].cpu().numpy()),
            "batch loss":loss.item(),
                  })
        ###########
        
    print("Epoch train loss:", np.mean(loss_history))
    print("Epoch valid loss:", validation())
        
    #get_ipython().system('mkdir -p trainedELMo')
    os.makedirs(outdir,exist_ok=True)
    torch.save(model, outdir+"Model"+str(e))
