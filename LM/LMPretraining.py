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

from NeuralLM import LanguageModel
from dataset import *
from pytransformer import LabelSmoothingLoss

BOS = "<S>"
EOS = "</S>"
UNK = "<unk>"
PAD = "<pad>"

use_wandb = True


batch_size = 175
batch_size_inf = 200
start = 31
epochs = 50

data_dir = "data-fixed/"
outdir = "trainedLM512-CE/"
preload = outdir + "LM-check"
vocab = json.load(open(data_dir+"vocab.json", "r"))
vocab_size = len(vocab)
training_set = PretrainDataset(data_dir+"train_seq.json", 5, 50, vocab[PAD]) #train_seq
validation_set = PretrainDataset(data_dir+"valid_seq.json", 5, 50, vocab[PAD])


training_generator = Loader(training_set, batch_size=batch_size, shuffle=True)
validation_generator = Loader(validation_set, batch_size=batch_size_inf, shuffle=False)
total_train = int(math.ceil(training_set.size / batch_size))
total_valid = int(math.ceil(validation_set.size / batch_size_inf))


device = torch.device("cuda")
model = LanguageModel(vocab, emb_dim=512, hidden_dim=512, dropout=0.1, emb_share=True, use_position=True, AdvSoft=5e-3).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab[PAD], reduction="mean").to(device)
# criterion = LabelSmoothingLoss(label_smoothing=0.1, tgt_vocab_size=vocab_size, ignore_index=vocab[PAD], reduction="batchmean").to(device)

lr = 1e-3
w_decay = 1e-6
optimizer = AdaBound(model.parameters(), lr=lr, final_lr=0.1, weight_decay=w_decay)


if use_wandb:
    import wandb

    wandb.init(entity="george0828zhang", project="contextual-matching-policy-gradient")
    wandb.config.update({
        "batch_size": batch_size,
        "learning rate":lr,
        "weight decay":w_decay
        })
    # wandb.watch([model])


def validation():
    model.eval()
    total_loss = []
    with torch.no_grad():   
        trange = tqdm(validation_generator, total=total_valid)
        for src in trange:
            src = src.to(device)
            
            logits = model(src[:,:-1])
            tgt = src[:,1:].contiguous()
            loss = criterion(logits.view(-1, vocab_size), tgt.view(-1))#.mean()

            total_loss.append(loss.item())
            
    return np.mean(total_loss)


vocab_inv = {a:b for b,a in vocab.items()}
def id2sent(ids):
    toks = (vocab_inv[i] for i in ids)
    return " ".join(toks)
def tstring(reward):
    return ", ".join([format(f, ".5g") for f in reward.cpu().numpy()])
def probs_ppx(logp, trg):
    # expects (len, vocab) & (len)
    with torch.no_grad():
        CE = F.cross_entropy(logp, trg, reduction='none')
        y0_probs = (-CE).exp() # probs for first sentence
        perplex = CE.mean().exp().item()
    return y0_probs, perplex




if preload != None:
    smodel = torch.load(preload)
    model.load_state_dict(smodel.state_dict())



for e in range(start, epochs+1):
    model.train()
    loss_history = []
    trange = tqdm(training_generator, total=total_train, desc="epoch {}".format(e))
    
    for i, src in enumerate(trange):
        src = src.to(device)
        
        logits = model(src[:,:-1])
        tgt = src[:,1:].contiguous()
        loss = criterion(logits.view(-1, vocab_size), tgt.view(-1))

        # y0_probs = (-loss[:tgt.shape[1]]).detach().exp() # probs for first sentence
        
        # loss = loss.mean()
        # perplex = loss.exp().item()
        y0_probs, perplex = probs_ppx(logits[0], tgt[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        loss_history.append(loss.item())
        trange.set_postfix(
            loss='{:.5g}'.format(loss.item()), 
            ppx='{:.5g}'.format(perplex))
        
        # M = torch.distributions.Categorical(logits=logits)
        # ys = M.sample()
        ys = torch.argmax(logits, dim=-1)
        
        if use_wandb:
            ### logging        
            wandb.log({     
            # print({
                "input":id2sent(src[0].cpu().numpy()),
                "output":id2sent(ys[0].cpu().numpy()),
                "target":id2sent(tgt[0].cpu().numpy()),
                "tarprob":tstring(y0_probs),
                "perplexity":min(perplex, 1e4),
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
