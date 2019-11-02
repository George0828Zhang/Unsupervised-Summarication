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
from dataset import *
from CM_VAE.ytvae import YT_VAE
# BOS = "<S>"
# EOS = "</S>"
# UNK = "<unk>"
# PAD = "<pad>"
BOS = "<|endoftext|>"
EOS = "<|endoftext|>"
UNK = "<|endoftext|>"
PAD = "<|endoftext|>"

use_wandb = True
device = torch.device("cuda")

data_dir = "/hdd/data-giga-gpt2-withEOF/"
train_path = data_dir + "train_seq.json"
valid_path = data_dir + "valid_seq.json"
vocab_path = data_dir + "vocab.json"

preload = None #"trained/PG-e2"
preload_LM = None #"trainedLM512-CE/LM-check"


start = 1
epochs = 20


vocab = json.load(open(vocab_path))
VOCAB_SIZE = len(vocab)
INPUT_LEN = 50
OUTPUT_LEN = 20


training_set = Dataset(train_path, INPUT_LEN, OUTPUT_LEN, vocab[PAD], unpaired=True) #train_seq
validation_set = Dataset(valid_path, INPUT_LEN, OUTPUT_LEN, vocab[PAD], unpaired=True)


batch_size = 120
batch_size_inf = batch_size
training_generator = Loader(training_set, batch_size=batch_size, shuffle=False)
validation_generator = Loader(validation_set, batch_size=batch_size_inf, shuffle=False)
total_train = int(math.ceil(training_set.size / batch_size))
total_valid = int(math.ceil(validation_set.size / batch_size_inf))

model = YT_VAE(VOCAB_SIZE, latent_size=1024, emb_tied=True, dropout=0.1, 
            d_model=256, nhead=8, num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=512, activation='relu')
model = model.to(device)
eps = 1e-10
learning_rate = 1e-4
weight_decay = 1e-6
aux_weight = 0.2

optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


######### eval perplex
def perplex(ys, log=True):
    with torch.no_grad():
        xent = evaluator.inference(ys[:,1:], start_index=vocab[BOS], return_prob=False)
    _ppx = xent.mean() if log else xent.mean().exp()
    return xent, _ppx.item()

if use_wandb:
    import wandb

    wandb.init(project="pretrained-VAE")
    wandb.config.update({
        "batch size": batch_size,
        "input len":INPUT_LEN,
        "summary len":OUTPUT_LEN,
        # "n_steps_backprop":n_steps_backprop,        
        "weight decay": weight_decay,
        "learning rate": learning_rate,
        "aux_weight": aux_weight,
        })
    # wandb.watch([translator, matcher])

# vocab_inv = {a:b for b,a in vocab.items()}
# def id2sent(ids):
#     toks = (vocab_inv[i] for i in ids)
#     return " ".join(toks)
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
def id2sent(ids):
    return tokenizer.decode(ids)
def tstring(reward):
    return ", ".join([format(f, ".5g") for f in reward.cpu().numpy()])

def criterion(logits, target):
    logits = logits.view(-1, VOCAB_SIZE)
    target = target.contiguous().view(-1)
    loss = F.nll_loss(logits, target, ignore_index = vocab[PAD], reduction = 'none')
    return loss.mean()

for e in range(start, epochs+1):
    bigbar = tqdm(training_generator, total=total_train, desc="[epoch] {}".format(e))
    
    for i, (src, tgt) in enumerate(bigbar):
        src = src.to(device)
        en_input = de_input = src[:,:-1]
        de_tgt = src[:, 1:]

        logp, kld = model(src=en_input, 
                                    src_pad_mask=(en_input == vocab[PAD]),
                                    tgt=de_input,
                                    tgt_pad_mask=(de_input == vocab[PAD]),
                                    )
        loss = criterion(logp, de_tgt)
        total_loss = loss + kld

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        ys = logp.argmax(dim = -1)

        bigbar.set_postfix(
            loss=loss.item(), 
            kld=kld.item(),
            )

        info = {
                "input":id2sent(en_input[0].cpu().numpy()),
                "output":id2sent(ys[0].cpu().numpy()),
                "target":id2sent(de_tgt[0].cpu().numpy()),
                "loss":loss.item(),
                "kld":kld.item(),
                "perplexity":min(loss.exp().item(), 1e5),        
        }

        if use_wandb: 
            wandb.log(info)
        else:
            print(info)
        ###########
        

        if i % 5000 == 4999:
            os.makedirs("trained_VAE",exist_ok=True)
            torch.save(model.state_dict(), "trained_VAE/VAE-check")        
            
    os.makedirs("trained_VAE",exist_ok=True)
    torch.save(model.state_dict(), "trained_VAE/VAE-e"+str(e))

