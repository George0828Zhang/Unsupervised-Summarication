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

        self.CE = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, word_ids):
        emb = self.position(self.embed(word_ids))
        out, (h, c) = self.lstm(emb)
        proj = F.linear(out, self.embed.weight) if self.emb_share else self.project(out)
        return proj

    def inference(self, sent):
        # (batch, len)
        batch_size, seqlen = sent.shape[:2]
        src = torch.ones(batch_size, 1).fill_(self.vocab[BOS]).type_as(sent.data)
        src = torch.cat((src, sent[:,:-1]), 1)
        tgt = sent.contiguous()
        
        logits = self.forward(src) # (1, len, vocab)
        
        CE = self.CE(logits.view(-1, self.vocab_size), tgt.view(-1))
        probs = (-CE).exp()
        return probs.view(batch_size, seqlen)
