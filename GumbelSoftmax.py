#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from architecture import *
from pytransformer import FullTransformer
from pointergenerator import PointerGenerator
from NeuralLM import LanguageModel
from dataset import *
BOS = "<S>"
EOS = "</S>"
UNK = "<unk>"
PAD = "<pad>"

use_wandb = True
device = torch.device("cuda")

data_dir = "data-fixed/"
train_path = data_dir + "train_seq.json"
valid_path = data_dir + "valid_seq.json"
vocab_path = data_dir + "vocab.json"
# embed_path = data_dir + "embeddings.npy"
# lm_path = data_dir + "trainedLM13"
# elmo_path = data_dir + "pretrain_ELMo"
preload = "trained/PG-e2" #data_dir + "Pretrain114999"
preload_LM = "trainedLM512-CE/LM-check"
# cached_map = data_dir + "candidate_map"


start = 3
epochs = 20


vocab = json.load(open(vocab_path))
VOCAB_SIZE = len(vocab)
INPUT_LEN = 50
OUTPUT_LEN = 20


training_set = Dataset(train_path, INPUT_LEN, OUTPUT_LEN, vocab[PAD]) #train_seq
validation_set = Dataset(valid_path, INPUT_LEN, OUTPUT_LEN, vocab[PAD])


batch_size = 64
batch_size_inf = batch_size
training_generator = Loader(training_set, batch_size=batch_size, shuffle=False)
validation_generator = Loader(validation_set, batch_size=batch_size_inf, shuffle=False)
total_train = int(math.ceil(training_set.size / batch_size))
total_valid = int(math.ceil(validation_set.size / batch_size_inf))

# translator = FullTransformer(
#     vocab_size=VOCAB_SIZE, emb_tied=True,
#     d_model=256, nhead=8, 
#     num_encoder_layers=2, num_decoder_layers=2, 
#     dim_feedforward=512, dropout=0.1, activation='relu')
translator = PointerGenerator(vocab_size=VOCAB_SIZE, d_model=256, d_emb=256, nhead=8, num_layers=2, dropout=0.1, coverage=True)
discriminator1 = LanguageModel(vocab, emb_dim=512, hidden_dim=512, dropout=0.4, emb_share=True, use_position=True)
evaluator = LanguageModel(vocab, emb_dim=512, hidden_dim=512, dropout=0.4, emb_share=True, use_position=True)
# remember to initialize

# preload
if preload != None:
    tmp = torch.load(preload)
    translator.load_state_dict(tmp)
    del tmp
if preload_LM != None:
    tmp = torch.load(preload_LM, map_location=lambda s,l: s)
    discriminator1.load_state_dict(tmp.state_dict())
    evaluator.load_state_dict(tmp.state_dict())
    del tmp

# send to device
translator.to(device)
discriminator1.to(device)
evaluator.to(device)
evaluator.eval()

eps = 1e-10
learning_rate = {'G':1e-4, 'D':1e-6}
weight_decay = {'G':1e-6, 'D':1e-4}
# learning_rate = {'G':1e-4, 'D':1e-5}
# weight_decay = {'G':1e-6, 'D':5e-5}
optimizer_G = torch.optim.RMSprop(translator.parameters(), lr=learning_rate['G'], weight_decay=weight_decay['G'])
optimizer_D = torch.optim.RMSprop(discriminator1.parameters(), lr=learning_rate['D'], weight_decay=weight_decay['D'])


######### eval perplex
def perplex(ys, log=True):
    with torch.no_grad():
        xent = evaluator.inference(ys[:,1:], start_index=vocab[BOS], return_prob=False)
    _ppx = xent.mean() if log else xent.mean().exp()
    return xent, _ppx.item()

######### training strategy
def train_D(src, tgt, gumbel_tau, N=1, update=True):
    translator.eval()
    discriminator1.train()

    real_loss_avg=[]
    fake_loss_avg=[]
    for _ in range(N):
        with torch.no_grad():
            # True values are positions that should be masked with float(‘-inf’) and False values will be unchanged. 
            src_mask = (src == vocab[PAD])
            ys, _ = translator(src, src_mask=src_mask, max_len=tgt.size(1), 
                start_symbol=vocab[BOS], gumbel_tau=gumbel_tau, return_index=True, keep_bos=False)

        fake_loss = discriminator1.inference(ys, start_index=vocab[BOS], return_prob=False).mean()
        real_loss = discriminator1.inference(tgt[:,1:], start_index=vocab[BOS], return_prob=False).mean()

        dloss = (real_loss - fake_loss)
        dloss.backward()

        fake_loss_avg.append(fake_loss.item())
        real_loss_avg.append(real_loss.item())

    if update:
        optimizer_D.step()
        optimizer_D.zero_grad()

    return np.mean(fake_loss_avg), np.mean(real_loss_avg)

# def CatXEnt(pred, target):
#     return -(target * torch.log(pred)).sum(dim=1).mean()

def train_G(src, max_len, gumbel_tau, N=1, update=True):
    translator.train()
    discriminator1.train()

    src_mask = (src == vocab[PAD]) 
    ys_hot, covloss = translator(src, src_mask=src_mask, max_len=max_len, 
            start_symbol=vocab[BOS], gumbel_tau=gumbel_tau, return_index=False, keep_bos=True)

    log_p_LM = discriminator1(ys_hot)#.detach()
    
    # (batch, len, vocab)
    # xent = (-ys_hot*log_p_LM).sum(-1)
    b,s,v = ys_hot.shape
    ys_hot = ys_hot.contiguous()
    xent = -torch.matmul(ys_hot.view(b*s, 1, v), log_p_LM.view(b*s, v, 1)) # (b*s, 1)
    xent = xent.view(b, s)

    # ys_hot = translator(src, src_mask=src_mask, max_len=max_len, 
    #         start_symbol=vocab[BOS], gumbel_tau=gumbel_tau, return_index=False, keep_bos=False)
    # b,s,v = ys_hot.shape
    # ys_hot = ys_hot.contiguous()
    # log_p = torch.log(ys_hot+eps)
    # xent = F.nll_loss(log_p.view(b*s, v), tgt.view(b*s), reduction='none').view(b, s)

    # (batch, len)
    gloss = xent.mean() + covloss

    gloss.backward()

    if update:
        optimizer_G.step()
        optimizer_G.zero_grad()
        optimizer_D.zero_grad() # clear gradient computed for G

    return gloss.item(), covloss.item(), xent.detach(), ys_hot.argmax(dim=-1)



if use_wandb:
    import wandb

    wandb.init(project="contextual-matching-policy-gradient")
    wandb.config.update({
        "batch size": batch_size,
        "input len":INPUT_LEN,
        "summary len":OUTPUT_LEN,
        # "n_steps_backprop":n_steps_backprop,        
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

gumbel_tau = 1.
baseline = 0.

for e in range(start, epochs+1):
    gumbel_tau = max(1e-3, 2**(1-start))
    bigbar = tqdm(training_generator, total=total_train, desc="[epoch] {}".format(e))
    
    for i, (src, tgt) in enumerate(bigbar):
        src = src.to(device)
        tgt = tgt.to(device)

        fake_loss, real_loss = train_D(src, tgt, gumbel_tau, N=5, update=True)
        g_loss, cov_loss, XEnt, ys = train_G(src, tgt.size(1), gumbel_tau, N=1, update=True)
        
        # log_perplex = XEnt.mean().item()

        XEnt, ppx = perplex(ys[0:], log=True)

        fake_loss = g_loss
        ### logging
        bigbar.set_postfix(
            fake_loss=fake_loss, 
            real_loss=real_loss,
            cov_loss=cov_loss,
            tau=gumbel_tau,
            lgppx=ppx,
            )

        info = {
                "input":id2sent(src[0].cpu().numpy()),
                "output":id2sent(ys[0].cpu().numpy()),
                "target":id2sent(tgt[0].cpu().numpy()),
                "fake loss":fake_loss,
                "real loss":real_loss, 
                "cov loss":cov_loss,
                "XEnt":tstring(XEnt[0]),
                "log perplexity":min(ppx, 1e5),
                      }
        if use_wandb: 
            wandb.log(info)
        else:
            print(info)
        ###########
        

        if i % 5000 == 4999:
            os.makedirs("trained",exist_ok=True)
            torch.save(translator.state_dict(), "trained/PG-check")        
            
    torch.save(translator.state_dict(), "trained/PG-e"+str(e))

