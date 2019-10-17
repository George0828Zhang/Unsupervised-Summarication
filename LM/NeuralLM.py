import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import random

from preprocessors import BOS, EOS, PAD, UNK # special tokens
import math
from tqdm.auto import tqdm
from transformer_nb2 import PositionalEncoding

class LanguageModel(nn.Module):
    def __init__(self, vocab, emb_dim, hidden_dim, dropout, emb_share=True, use_position=True):
        super().__init__()        
        self.vocab = vocab
        self.vocab_size = len(vocab)
                
        self.embed = nn.Embedding(self.vocab_size, emb_dim)
        self.position = PositionalEncoding(emb_dim, dropout) if use_position else nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=2, dropout=dropout, batch_first=True, bidirectional=False)
        self.emb_share = emb_share
        if not emb_share:
            self.project = nn.Linear(hidden_dim, self.vocab_size) 
    
    def forward(self, word_ids):
        emb = self.position(self.embed(word_ids))
        out, (h, c) = self.lstm(emb)
        proj = F.linear(out, self.embed.weight) if self.emb_share else self.project(out)
        return F.log_softmax(proj, dim=-1)

    def decode(self, src, max_len, mode='sample'):
        batch_size = src.size(0)
        word_ids = src[:,:1] # should be BOS (batch, 1)
        logits = []

        for i in range(max_len):
            emb = self.position(self.embed(word_ids[:,-1:])) # (batch, 1, emb)
            out, (h, c) = self.lstm(emb, None if i == 0 else (h, c)) # (batch, 1, hidden)
            proj = F.linear(out, self.embed.weight) if self.emb_share else self.project(out)
            proj = F.log_softmax(proj, dim=-1) # (batch, 1, vocab)

            if mode == 'argmax':
                values, next_words = torch.max(proj, dim=-1, keepdim=True)
            elif mode == 'sample':
                m = torch.distributions.Categorical(logits=proj)
                next_words = m.sample()
            else:
                raise

            logits.append(proj)
            word_ids = torch.cat((word_ids, next_words), dim=1)
        logits = torch.cat(logits, dim=1)
        return word_ids[:,1:], logits



    def inference(self, sent):
        # (batch, len)
        batch_size, seqlen = sent.shape[:2]
        src = torch.ones(batch_size, 1).fill_(self.vocab[BOS]).type_as(sent.data)
        src = torch.cat((src, sent[:,:-1]), 1)
        tgt = sent.contiguous()
        
        logits = self.forward(src) # (1, len, vocab)
            
        CE = F.cross_entropy(logits.view(-1, self.vocab_size), tgt.view(-1), reduction='none')
        probs = (-CE).exp()
        return probs.view(batch_size, seqlen)
