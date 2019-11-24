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
from pytransformer import FullTransformer
from pointergenerator import PointerGenerator
from NeuralLM import LanguageModel, GPT2LM
from dataset import *
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

data_dir = "data-giga-gpt2-eos/"
train_path = data_dir + "train_seq.json"
valid_path = data_dir + "valid_seq.json"
vocab_path = data_dir + "vocab.json"

preload = None #"trained/PG-check"
preload_LM = None #"trainedLM512-CE/LM-check"


start = 1
epochs = 20


vocab = json.load(open(vocab_path))
VOCAB_SIZE = len(vocab)
INPUT_LEN = 50
OUTPUT_LEN = 20


training_set = Dataset(train_path, INPUT_LEN, OUTPUT_LEN, vocab[PAD], unpaired=True) #train_seq
validation_set = Dataset(valid_path, INPUT_LEN, OUTPUT_LEN, vocab[PAD], unpaired=True)


batch_size = 32
batch_size_inf = batch_size
training_generator = Loader(training_set, batch_size=batch_size, shuffle=True)
validation_generator = Loader(validation_set, batch_size=batch_size_inf, shuffle=False)
total_train = int(math.ceil(training_set.size / batch_size))
total_valid = int(math.ceil(validation_set.size / batch_size_inf))

translator = PointerGenerator(vocab_size=VOCAB_SIZE, d_model=256, d_emb=256, nhead=8, num_layers=2, dropout=0.1, coverage=True)
discriminator1 = LanguageModel(vocab, emb_dim=512, hidden_dim=512, dropout=0.4, emb_share=True, use_position=True)
# discriminator1 = GPT2LM()
# evaluator = LanguageModel(vocab, emb_dim=512, hidden_dim=512, dropout=0.4, emb_share=True, use_position=True)
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
# evaluator.to(device)
# evaluator.eval()

eps = 1e-10
learning_rate = {'G':1e-4, 'D':1e-4}
# weight_decay = {'G':1e-6, 'D':1e-6}
weight_decay = {'G':None, 'D':None}
fake_weight = 0 # 0.05
wgan_type = "none"
wgan_clip = 0.05
gp_weight = 10
# optimizer_G = torch.optim.RMSprop(translator.parameters(), lr=learning_rate['G'], weight_decay=weight_decay['G'])
# optimizer_D = torch.optim.RMSprop(discriminator1.parameters(), lr=learning_rate['D'], weight_decay=weight_decay['D'])
optimizer_G = torch.optim.RMSprop(translator.parameters(), lr=learning_rate['G'])
optimizer_D = torch.optim.RMSprop(discriminator1.parameters(), lr=learning_rate['D'])


######### eval perplex
def perplex(evaluator, ys, log=True):
    with torch.no_grad():
        xent = evaluator.inference(ys[:,1:], start_index=vocab[BOS], return_prob=False)
    _ppx = xent.mean() if log else xent.mean().exp()
    return xent, _ppx.item()

######### training strategy
def hotify(seq):
    if seq.dim() == 3:
        return seq
    one_hot = torch.zeros((seq.size(0),seq.size(1),VOCAB_SIZE)).to(seq.device)
    one_hot.scatter_(2, seq.unsqueeze(2), 1.0)
    return one_hot

def compute_gp(D,real_seq,fake_seq):
    bs = int(min(real_seq.size(0),fake_seq.size(0)))
    seq_len = int(min(real_seq.size(1),fake_seq.size(1)))
    dim = int(real_seq.size(2))
    real_seq = real_seq[:bs,:seq_len,:]
    fake_seq = fake_seq[:bs,:seq_len,:]
    
    # Interpolation
    alpha = torch.rand(bs, 1).unsqueeze(2).expand((-1,seq_len,dim)).to(real_seq.device)
    inters = alpha * real_seq + ((1 - alpha) * fake_seq)
    inter = inters.detach()
    inter.requires_grad = True
        
    score_inter = D(inter)
    gradients = torch.autograd.grad(outputs=score_inter, inputs=inter,
                    grad_outputs=torch.ones(score_inter.size()).to(real_seq.device),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_D(src, tgt, gumbel_tau, N=1, update=True):
    translator.eval()
    discriminator1.train()

    real_loss_avg=[]
    fake_loss_avg=[]
    gp_loss_avg=[]
    for _ in range(N):
        with torch.no_grad():
            # True values are positions that should be masked with float(‘-inf’) and False values will be unchanged. 
            src_mask = (src == vocab[PAD])
            ys, _ = translator(src, src_mask=src_mask, max_len=tgt.size(1), 
                start_symbol=vocab[BOS], gumbel_tau=gumbel_tau, return_index=False, keep_bos=False)

        fake_loss = discriminator1.inference(ys.argmax(-1), start_index=vocab[BOS], ignore_index=vocab[PAD], return_prob=False).mean()
        real_loss = discriminator1.inference(tgt[:,1:], start_index=vocab[BOS], ignore_index=vocab[PAD], return_prob=False).mean()
        
        dloss = real_loss - fake_loss*fake_weight

        if wgan_type=="gp":
            gp_loss = compute_gp(discriminator1, real_seq=hotify(tgt[:,1:]), fake_seq=hotify(ys))
            dloss += gp_weight*gp_loss
            gp_loss_avg.append(gp_loss.item())

        dloss /= N
        dloss.backward()

        fake_loss_avg.append(fake_loss.item())
        real_loss_avg.append(real_loss.item())
        

    if update:
        optimizer_D.step()
        optimizer_D.zero_grad()

        if wgan_type=="clip":
            for p in discriminator1.parameters():
                p.data.clamp_(-wgan_clip, wgan_clip)

    outputs = np.mean(fake_loss_avg), np.mean(real_loss_avg)
    if wgan_type=="gp":
        outputs = outputs + (np.mean(gp_loss_avg),)
    return outputs

def train_G(src, max_len, gumbel_tau, N=1, update=True, lm_path_grad=True):
    translator.train()
    discriminator1.train()

    for _ in range(N):
        src_mask = (src == vocab[PAD]) 
        ys_hot, covloss = translator(src, src_mask=src_mask, max_len=max_len, 
                start_symbol=vocab[BOS], gumbel_tau=gumbel_tau, return_index=False, keep_bos=True)

        if lm_path_grad:
            # use 0:-1
            lm_input = ys_hot[:,:-1] #.argmax(dim=-1)
            lm_logits = discriminator1(lm_input)
            log_p_LM = F.log_softmax(lm_logits, dim=-1)
            # gets 1:end
        else:
            with torch.no_grad():
                # use 0:-1
                lm_input = ys_hot[:,:-1].argmax(dim=-1)
                lm_logits = discriminator1(lm_input)
                log_p_LM = F.log_softmax(lm_logits, dim=-1)
                # gets 1:end

        # use 1:end
        xent_input = ys_hot[:,1:].contiguous()
        b,s,v = xent_input.shape
        xent = F.kl_div(log_p_LM.view(b*s, v), xent_input.view(b*s, v), reduction="batchmean") # (b, s)

        gloss = xent + covloss

        ######### Contextual Matching
        # x_reps = discriminator1.embed(src) # (batch, xlen, emb)
        # y_reps = torch.matmul(xent_input, discriminator1.embed.weight) # (batch, ylen, emb)

        # x_reps = F.normalize(x_reps, p=2, dim=-1) # (batch, xlen, emb)
        # y_reps = F.normalize(y_reps, p=2, dim=-1) # (batch, ylen, emb)
        # cosine = torch.matmul(y_reps, x_reps.transpose(-2, -1)) # (batch, ylen, xlen)
        # cosine, _ = torch.max(cosine, dim=-1) # (batch, ylen)
        # cm_loss = (1 - cosine).mean()
        # gloss += cm_loss
        #############################

        gloss /= N
        gloss.backward()
        # (batch, len)
        #gloss = (xent.mean() + covloss)/N
        #gloss.backward()

    if update:
        optimizer_G.step()
        optimizer_G.zero_grad()
        optimizer_D.zero_grad() # clear gradient computed for G

    return gloss.item(), covloss.item(), ys_hot.argmax(dim=-1)



if use_wandb:
    import wandb

    wandb.init(project="contextual-matching-policy-gradient")
    wandb.config.update({
        "batch size": batch_size,
        "input len":INPUT_LEN,
        "summary len":OUTPUT_LEN,
        # "n_steps_backprop":n_steps_backprop,        
        "weight decay": weight_decay,
        "learning rate": learning_rate,
        "fake wieght": fake_weight,
        "gp weight": gp_weight,
        "wgan clip": wgan_clip,
        "wgan type": wgan_type,
        })
    # wandb.watch([translator, matcher])

# vocab_inv = {a:b for b,a in vocab.items()}
# def id2sent(ids):
#     toks = (vocab_inv[i] for i in ids)
#     return " ".join(toks)
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
def id2sent(ids):
    return tokenizer.decode(ids)
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
        g_loss, cov_loss, ys = train_G(src, tgt.size(1), gumbel_tau, N=1, update=True, lm_path_grad=True)
        
        # fake_loss = g_loss
        ### logging
        bigbar.set_postfix(
            fake_loss=fake_loss, 
            real_loss=real_loss,
            # cov_loss=cov_loss,
            tau=gumbel_tau,
            )

        info = {
                "input":id2sent(src[0].cpu().numpy()),
                "output":id2sent(ys[0].cpu().numpy()),
                "target":id2sent(tgt[0].cpu().numpy()),
                "fake loss":fake_loss,
                "real loss":real_loss, 
                "cov loss":cov_loss,
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

