#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
import math
from tqdm.auto import tqdm, trange
from adabound import AdaBound

from preprocessors import BOS, EOS, PAD, UNK
from ELMo import LanguageModel
from dataset import *
from transformer_nb2 import LabelSmoothing

# In[2]:

## try to use EOS instead of PAD


batch_size = 256
batch_size_inf = 256


# In[3]:





# In[4]:


data_dir = "data-20k/"
outdir = "trainedELMo/"
vocab = json.load(open(data_dir+"vocab.json", "r"))
vocab_size = len(vocab)
training_set = PretrainDataset(data_dir+"train_seq.json", 20, 20, vocab[EOS]) #train_seq
validation_set = PretrainDataset(data_dir+"valid_seq.json", 20, 20, vocab[EOS])


# In[5]:


training_generator = Loader(training_set, batch_size=batch_size, shuffle=True)
validation_generator = Loader(validation_set, batch_size=batch_size_inf, shuffle=False)
total_train = int(math.ceil(training_set.size / batch_size))
total_valid = int(math.ceil(validation_set.size / batch_size_inf))


# In[6]:


device = torch.device("cuda")
model = LanguageModel(vocab, emb_dim=1024, hidden_dim=1024, dropout=0.1, emb_share=True).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab[PAD], reduction="none").to(device)
# crit = LabelSmoothing(size=vocab_size, padding_idx=vocab[PAD], smoothing=0.1).to(device)
# def criterion(x,y):
#     x = F.log_softmax(x, dim=-1)
#     n_token = (y != vocab[PAD]).data.sum().item()
#     # n_token = y.shape[0]
#     return crit(x, y)/n_token
# optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
lr = 1e-3
w_decay = 1e-6
optimizer = AdaBound(model.parameters(), lr=lr, final_lr=0.1, weight_decay=w_decay)

# In[7]:

import wandb

wandb.init(entity="george0828zhang", project="contextual-matching-policy-gradient")
wandb.config.update({
    "batch_size": batch_size,
    "learning rate":lr,
    "weight decay":w_decay
    })
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
            loss = criterion(logits.view(-1, vocab_size), tgt.view(-1)).mean()

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
epochs = 20


# In[11]:


# if start != 1:
#     smodel = torch.load("trainedELMo/Model"+str(start-1))
#     model.load_state_dict(smodel['model'])


# In[12]:


for e in range(start, epochs+1):
    model.train()
    loss_history = []
    trange = tqdm(training_generator, total=total_train, desc="epoch {}".format(e))
    
    for i,(src, tgt) in enumerate(trange):
        src = src.to(device)
        tgt = tgt.to(device)
        
        logits = model(src)
        loss = criterion(logits.view(-1, vocab_size), tgt.view(-1))

        y0_probs = (-loss[:tgt.shape[1]]).detach().exp() # probs for first sentence
        
        loss = loss.mean()
        perplex = loss.exp().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        loss_history.append(loss.item())
        trange.set_postfix(**{'loss':'{:.5f}'.format(loss.item())})
        
        M = torch.distributions.Categorical(logits=logits)
        ys = M.sample()
        
        ### logging        
        wandb.log({     
        # print({
            "input":id2sent(src[0].cpu().numpy()),
            "output":id2sent(ys[0].cpu().numpy()),
            "target":id2sent(tgt[0].cpu().numpy()),
            "tarprob":tstring(y0_probs),
            "perplexity":perplex,
            "batch loss":loss.item(),            
                  })
        ###########

        if i % 5000 == 4999:
            os.makedirs(outdir,exist_ok=True)
            torch.save(model, outdir+"LM-check")
        
    print("Epoch loss (train, valid):", np.mean(loss_history), validation())
        
    #get_ipython().system('mkdir -p trainedELMo')
    os.makedirs(outdir,exist_ok=True)
    torch.save(model, outdir+"LM"+str(e))
