import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import random

from pytransformer import PositionalEncoding

class LanguageModel(nn.Module):
    def __init__(self, vocab, emb_dim, hidden_dim, dropout, emb_share=True, use_position=True):
        super().__init__()        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.emb_share = emb_share

        self.embed = nn.Embedding(self.vocab_size, emb_dim)
        self.position = PositionalEncoding(emb_dim, dropout) if use_position else nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=2, dropout=dropout, batch_first=True, bidirectional=False)
        
        if not emb_share:
            self._project = nn.Linear(hidden_dim, self.vocab_size) 

    def project(self, h):
        # (batch, len, hidden)
        if self.emb_share:
            proj = F.linear(h, self.embed.weight)
        else:
            proj = self._project(h)
        return proj
   
    def decode(self, src, max_len, mode='sample'):
        batch_size = src.size(0)
        word_ids = src[:,:1] # should be BOS (batch, 1)
        logits = []

        for i in range(max_len):
            emb = self.position(self.embed(word_ids[:,-1:])) # (batch, 1, emb)
            out, (h, c) = self.lstm(emb, None if i == 0 else (h, c)) # (batch, 1, hidden)
            proj = self.project(out)
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


    def forward(self, word_ids):
        if word_ids.dim() < 2:
            raise RuntimeError("word_ids should be (batch, len) or (batch, len, vocab)")
        elif word_ids.dim() == 2:
            emb = self.position(self.embed(word_ids))
        elif word_ids.dim() == 3:
            emb = torch.matmul(word_ids, self.embed.weight)
            emb = self.position(emb)
        out, (h, c) = self.lstm(emb)

        proj = self.project(out)
        return proj #F.log_softmax(proj, dim=-1)
    
    def inference(self, sent, start_index, ignore_index=-100, return_prob=False):
        # (batch, len)
        batch_size, seqlen = sent.shape[:2]
        src = torch.ones(batch_size, 1).fill_(start_index).type_as(sent.data)
        src = torch.cat((src, sent[:,:-1]), 1)
        tgt = sent.contiguous()
        
        logits = self.forward(src) # (1, len, vocab)
            
        CE = F.cross_entropy(logits.view(-1, self.vocab_size), tgt.view(-1), 
            ignore_index=ignore_index, reduction='none').view(batch_size, seqlen)

        if not return_prob:
            return CE
        else:
            return (-CE).exp()

from transformers import GPT2LMHeadModel, GPT2Tokenizer
class GPT2LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('distilgpt2')
        self.vocab_size = self.gpt2.config.vocab_size

    def forward(self, word_ids):
        logits, past = self.gpt2(word_ids)
        return logits #F.log_softmax(logits, dim=-1)

    def inference(self, sent, start_index, ignore_index=-100, return_prob=False):
        # (batch, len)
        batch_size, seqlen = sent.shape[:2]
        src = torch.ones(batch_size, 1).fill_(start_index).type_as(sent.data)
        src = torch.cat((src, sent[:,:-1]), 1)
        tgt = sent.contiguous()
      
        logits = self.forward(src) # (1, len, vocab)
                   
        CE = F.cross_entropy(logits.view(-1, self.vocab_size), tgt.view(-1), 
            ignore_index=ignore_index, reduction='none').view(batch_size, seqlen)

        if not return_prob:
            return CE
        else:
            return (-CE).exp()


