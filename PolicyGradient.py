#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture import *
from dataset import *


# In[2]:


data_dir = "data-giga/"
train_path = data_dir + "train_seq.json"
valid_path = data_dir + "valid_seq.json"
vocab_path = data_dir + "vocab.json"
embed_path = data_dir + "embeddings.npy"
lm_path = data_dir + "trainedLM13"
elmo_path = data_dir + "pretrain_ELMo"
preload = None #data_dir + "Pretrain114999"
cached_map = data_dir + "candidate_map"


# In[3]:


vocab = json.load(open(vocab_path))
VOCAB_SIZE = len(vocab)
INPUT_LEN = 50
OUTPUT_LEN = 20


# In[4]:


training_set = Dataset(train_path, INPUT_LEN, OUTPUT_LEN, vocab[PAD]) #train_seq
validation_set = Dataset(valid_path, INPUT_LEN, OUTPUT_LEN, vocab[PAD])


# In[5]:


batch_size = 100
batch_size_inf = 100
training_generator = Loader(training_set, batch_size=batch_size, shuffle=True)
validation_generator = Loader(validation_set, batch_size=batch_size_inf, shuffle=False)
total_train = int(math.ceil(training_set.size / batch_size))
total_valid = int(math.ceil(validation_set.size / batch_size_inf))


# In[6]:

candidate_map_cached = os.path.isfile(cached_map)
device = torch.device("cuda")
embeddings = torch.Tensor(np.load(embed_path)).to(device)
elmo = torch.load(elmo_path, map_location=lambda storage, location: storage)
LM = torch.load(lm_path, map_location=lambda storage, location: storage)
candidate_map = torch.load(cached_map) if candidate_map_cached else None
matcher = ContextMatcher(embeddings, elmo, LM, candidate_map).to(device)
if not candidate_map_cached:
    torch.save(matcher.candidate_map, cached_map)
matcher.eval()

# fix error for LM class change
LM.emb_share = False

# In[7]:


# translator = make_translator(
#     VOCAB_SIZE, VOCAB_SIZE, N=4, d_model=256,
#     d_ff=1024, h=8, dropout=0.1, emb_share=True).to(device)
translator = PointerGenerator(
    hidden_dim=128, emb_dim=128, input_len=INPUT_LEN, 
    output_len=OUTPUT_LEN, voc_size=VOCAB_SIZE, coverage=True, eps=1e-9).to(device)

if preload != None:
    tmp = torch.load(preload)
    translator.load_state_dict(tmp)

learning_rate = 1e-4
weight_decay = 1e-5
optimizer = torch.optim.RMSprop(translator.parameters(), lr=learning_rate, weight_decay=weight_decay)


# In[8]:


start = 1
epochs = 10
n_steps_backprop = 1
adjust_r = True
zero_mean_r = False
unit_standard_r = False

# In[ ]:


import wandb

wandb.init(project="contextual-matching-policy-gradient")
wandb.config.update({
    "batch_size": batch_size,
    "input len":INPUT_LEN,
    "summary len":OUTPUT_LEN,
    "n_steps_backprop":n_steps_backprop,
    "adjust":adjust_r,
    "zero mean":zero_mean_r,
    "unit standard":unit_standard_r,
    "weight decay": weight_decay,
    "learning rate": learning_rate
    })
# wandb.watch([translator, matcher])

vocab_inv = {a:b for b,a in vocab.items()}
def id2sent(ids):
    toks = (vocab_inv[i] for i in ids)
    return " ".join(toks)
def tstring(reward):
    return ", ".join([format(f, ".5g") for f in reward.cpu().numpy()])

# In[ ]:


for e in range(start, epochs+1):
    translator.train()
    # print("[epoch]", e)
    loss_history = []
    trange = tqdm(training_generator, total=total_train, desc="[epoch] {}".format(e))
    
    losses = []
    baseline = 0
    for i, (src, tgt) in enumerate(trange):
        src = src.to(device)
        
        src_mask = (src != vocab[PAD]).unsqueeze(-2)

        ys, log_p, covloss = translator(src=src, src_mask=src_mask, max_len=OUTPUT_LEN, start_symbol=vocab[BOS])
                
        reward, (cm, fm), baseline = rewards_compute(
            matcher=matcher,
            src=src, ys=ys, log_p=log_p, adjust=adjust_r, 
            zero_mean=zero_mean_r, unit_standard=unit_standard_r, baseline=baseline, gamma=0.99, eps=1e-9)
                
        loss = -reward.mean() + covloss
            
        ### logging        
        wandb.log({
            "input":id2sent(src[0].cpu().numpy()),
            "output":id2sent(ys[0].cpu().numpy()),
            "target":id2sent(tgt[0].cpu().numpy()),
            "reward cm":tstring(cm[0]), 
            "reward fm":tstring(fm[0]), 
            "batch loss":loss.item(),
            "coverage weight":translator.cov_weight.data.item(),
            "batch reward context":cm.sum(-1).mean().item(),
            "batch reward fluency":fm.sum(-1).mean().item(),
            "baseline":baseline,
                  })
        ###########
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # losses.append(loss)
        # if len(losses) >= n_steps_backprop:                 
        #     optimizer.zero_grad()
        #     sum(losses).backward()
        #     optimizer.step()
        #     losses = []
    
        loss_history.append(loss.item())
        trange.set_postfix(**{'loss':'{:.5f}'.format(loss.item())})

        if i % 5000 == 4999:
            os.makedirs("trained",exist_ok=True)
            torch.save(translator.state_dict(), "trained/PG-check")
        
        
    print("Epoch train loss:", np.mean(loss_history))
    
    torch.save(translator.state_dict(), "trained/PG-e"+str(e))

