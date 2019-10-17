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
from NeuralLM import LanguageModel
from dataset import *
from transformer_nb2 import LabelSmoothing, PositionalEncoding


## try to use EOS instead of PAD
use_wandb = True

batch_size = 88 # (464, 20) (88, 100)
batch_size_inf = batch_size

eps = 1e-10
lr = {'g':1e-5, 'd':5e-3}
w_decay = {'g':1e-6, 'd':1e-4}
train_iters = {'g':5, 'd':5}
gamma = 0.79
update_freq = 0.999
mode = "decode"


data_dir = "data-wiki103/"
outdir = "GANLM/"
vocab = json.load(open(data_dir+"vocab.json", "r"))
vocab_size = len(vocab)
training_set = PretrainDataset(data_dir+"train_seq.json", 7, 100, vocab[EOS]) #train_seq
validation_set = PretrainDataset(data_dir+"valid_seq.json", 7, 100, vocab[EOS])




training_generator = Loader(training_set, batch_size=batch_size, shuffle=False)
validation_generator = Loader(validation_set, batch_size=batch_size_inf, shuffle=False)
total_train = int(math.ceil(training_set.size / batch_size))
total_valid = int(math.ceil(validation_set.size / batch_size_inf))




device = torch.device("cuda")
model = LanguageModel(vocab, emb_dim=256, hidden_dim=256, dropout=0.1, emb_share=True).to(device)


### discriminator
class Discriminator(nn.Module):
    def __init__(self, vocab, emb_dim, hidden_dim, dropout, use_position=True):
        super().__init__()        
        self.vocab = vocab
        self.vocab_size = len(vocab)
                
        self.embed = nn.Embedding(self.vocab_size, emb_dim)
        self.position = PositionalEncoding(emb_dim, dropout) if use_position else nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=2, dropout=dropout, batch_first=True, bidirectional=False)
        self.project = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
            )

    
    def forward(self, word_ids):
        emb = self.position(self.embed(word_ids))
        out, (h, c) = self.lstm(emb)
        proj = self.project(out).squeeze(-1)
        return torch.sigmoid(proj)

D_model = Discriminator(vocab, emb_dim=128, hidden_dim=128, dropout=0.4).to(device)


# optimizer_g = torch.optim.Adam(model.parameters(), betas=(0.5,0), lr=lr['g'], weight_decay=w_decay['g'])
# optimizer_d = torch.optim.Adam(D_model.parameters(), betas=(0.5,0), lr=lr['d'], weight_decay=w_decay['d'])
optimizer_g = torch.optim.RMSprop(model.parameters(), lr=lr['g'], weight_decay=w_decay['g'])
optimizer_d = torch.optim.RMSprop(D_model.parameters(), lr=lr['d'], weight_decay=w_decay['d'])



# In[7]:

if use_wandb:
    import wandb

    wandb.init(entity="george0828zhang", project="contextual-matching-policy-gradient")
    wandb.config.update({
        "batch_size": batch_size,
        "learning rate":lr,
        "weight decay":w_decay,
        "gamma":gamma,
        "update freq":update_freq,
        "mode":mode,
        "train iters":train_iters
        })

# wandb.watch([model])



vocab_inv = {a:b for b,a in vocab.items()}
def id2sent(ids):
    toks = (vocab_inv[i] for i in ids)
    return " ".join(toks)
def tstring(reward):
    return ", ".join([format(f, ".5g") for f in reward.cpu().numpy()])


# In[10]:
def mean(l):
    return sum(l)/(len(l)+eps)

start = 1
epochs = 10

def train_D(N, src, tgt):
    model.eval()
    D_model.train()

    bar = range(N) #trange(N, desc="train D", leave=False)
    h_real_scores = []
    h_fake_scores = []
    h_D_loss = []

    optimizer_d.zero_grad()

    for i in bar:
        with torch.no_grad():            
            if mode == "decode":
                ys, logits = model.decode(src, max_len=tgt.size(1))
            else:
                logits = model(src)        
                M = torch.distributions.Categorical(logits=logits)
                ys = M.sample() # fake data

        # both (batch*len)
        real_score = D_model(tgt).view(-1)
        fake_score = D_model(ys).view(-1)

        # likelihood = torch.cat((real_score, fake_score), dim=0) # (2*batch*len)        
        # target = torch.cat((real_score*0+1, real_score*0), dim=0)
        # loss = F.binary_cross_entropy(likelihood, target, reduction='mean')
        all_ones = (real_score*0+1).detach()
        loss_real = F.binary_cross_entropy(real_score, all_ones, reduction='mean')
        loss_fake = -F.binary_cross_entropy(fake_score, all_ones, reduction='mean')

        loss = loss_real + loss_fake

        
        loss.backward()

        h_real_scores.append(real_score.mean().item())
        h_fake_scores.append(fake_score.mean().item())
        h_D_loss.append(loss.item())

    optimizer_d.step()
    return mean(h_real_scores), mean(h_fake_scores), mean(h_D_loss)

def discount_r(rewards, gamma):
    batch_size, max_len = rewards.shape[:2]
    rewards_adjust = []
    littleR = 0
    for t in reversed(range(max_len)):
        r = rewards[:,t] # (batch,)
        littleR = r + gamma*littleR
        rewards_adjust.insert(0, littleR)
    # (batch,)
    return torch.stack(rewards_adjust, dim=1)

def train_G(N, src, baseline, gamma):
    model.train()
    D_model.eval()

    bar = range(N) #trange(N, desc="train G", leave=False)
    h_G_loss = []
    h_reward = []
    optimizer_g.zero_grad()

    for i in bar:
        if mode == "decode":
            ys, logits = model.decode(src, max_len=src.size(1))
            M = torch.distributions.Categorical(logits=logits)
        else:
            logits = model(src)
            M = torch.distributions.Categorical(logits=logits)
            ys = M.sample() # (batch, len)

        with torch.no_grad():
            rewards = 2*D_model(ys) - 1 # (batch, len)

        rewards = discount_r(rewards, gamma)
        
        rewardTensor_based = rewards - baseline
        baseline = baseline*update_freq + rewards.mean().item()*(1-update_freq)   
      
        loss = -(M.log_prob(ys)*rewardTensor_based).sum()
        
        loss.backward()
        

        h_G_loss.append(loss.item())
        h_reward.append(rewards.mean().item())

    optimizer_g.step()

    return mean(h_reward), mean(h_G_loss), baseline

baseline = 0
for e in range(start, epochs+1):
    model.train()
    loss_history = []
    bigbar = tqdm(training_generator, total=total_train, desc="epoch {}".format(e))
    
    for i,src in enumerate(bigbar):
        src = src.to(device)
        # tgt = tgt.to(device)
        tgt = src[:,1:]
        src = src[:,:-1]

        # train_iters
        realsc, fakesc, d_loss = train_D(train_iters['d'], src=src, tgt=tgt)

        reward, g_loss, baseline = train_G(train_iters['g'], src=src, baseline=baseline, gamma=gamma)

        
        logits = model(src[:1]) # (1, len, vocab)
        M = torch.distributions.Categorical(logits=logits)
        ys = M.sample() # (1, len)
        y0_logp = M.log_prob(tgt[:1]).detach()
        ln_perplex = (-y0_logp.mean()).item()
        y0_probs = y0_logp.exp() # (1, len)

        bigbar.set_postfix(
            real=realsc, fake=fakesc, ln_perplex=ln_perplex
            # ,d_loss=d_loss, reward=reward, g_loss=g_loss, baseline=baseline,
        )


        if use_wandb:
            ### logging        
            wandb.log({   
                "input":id2sent(src[0].cpu().numpy()),
                "output":id2sent(ys[0].cpu().numpy()),
                "target":id2sent(tgt[0].cpu().numpy()),
                "tarprob":tstring(y0_probs[0]),
                "log perplexity":ln_perplex,
                "real score":realsc, 
                "fake score":fakesc,
                "D loss":d_loss,
                "G loss":g_loss, 
                "batch reward":reward, 
                "baseline":baseline,
                      })
            ###########        
        

        if i % 5000 == 4999:
            os.makedirs(outdir,exist_ok=True)
            torch.save(model, outdir+"LM-check")
        
    # print("Epoch loss (train, valid):", np.mean(loss_history), validation())
        
    #get_ipython().system('mkdir -p trainedELMo')
    os.makedirs(outdir,exist_ok=True)
    torch.save(model, outdir+"LM"+str(e))
