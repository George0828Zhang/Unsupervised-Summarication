import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.autograd import Variable
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2., dtype=torch.float32) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class FullTransformer(nn.Transformer):
    def __init__(self, vocab_size, emb_tied=False, dropout=0.5, **kwargs):
        super().__init__(dropout=dropout, **kwargs)
# self.d_model = d_model
#         self.nhead = nhead
        self.vocab_size = vocab_size
        self.emb_tied = emb_tied

        self.embeddings = nn.Embedding(vocab_size, self.d_model)
        self.positional = PositionalEncoding(self.d_model, dropout)

        if not emb_tied:
            self._generator = nn.Linear(self.d_model, vocab_size)

    def embed(self, x):
        if x.dim() < 2:
            raise RuntimeError("x should be (batch, len) or (batch, len, vocab)")
        elif x.dim() == 2:
            emb = self.embeddings(x)
        elif x.dim() == 3:
            emb = torch.matmul(x, self.embeddings.weight)
        return self.positional(emb)

    def generator(self, x):
        proj = F.linear(x, self.embeddings.weight) if self.emb_tied else self._generator(x)
        return F.log_softmax(proj, dim=-1)
        
    def forward(self, src, src_mask, max_len, start_symbol, gumbel_tau=1., return_index=False, keep_bos=False):
        batch_size = src.size(0)

        r"""
            Shapes for encoder:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`."""

        src_emb = self.embed(src) # (batch, xlen, emb)
        
        memory = self.encoder(src_emb.transpose(0, 1), src_key_padding_mask=src_mask)
        # (xlen, batch, emb)
        
        # ys = torch.ones(1, batch_size).fill_(start_symbol).type_as(src)
        # # (1, batch)

        # logits = []

        # for i in range(max_len):
        #   tgt_mask = self.generate_square_subsequent_mask(ys.size(1)) #.type_as(src_mask)
        #     out = self.decoder(self.embeddings(ys), memory, 
        #       tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
        #     # (i+1, batch, emb)

        #     log_probs = self.generator(out[-1,...]) # (batch, vocab)
        #     M = torch.distributions.Categorical(logits=log_probs)
        #     next_words = M.sample() # (batch, 1)

        #     ys = torch.cat((ys, next_words.unsqueeze(0)), dim=0)
            
        #     logits.append(log_probs)
        
        # logits = torch.stack(logits,dim=1) # (batch, max_len, vocab)

        # return ys[1:,:].transpose(0, 1), logits


        # use gumbel softmax instead

        # ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src)
        # # (batch, 1)
        # ys = torch.scatter_(dim=-1, index=ys.unsqueeze(0), value=1.) # one-hot

        # one-hot
        ys = torch.zeros((1, batch_size, self.vocab_size)).type_as(memory)
        ys[...,start_symbol] = 1.

        # (1, batch, vocab)
        # val, ind = ys.max(dim=-1)
        # assert((ind == start_symbol).all())
        
        for i in range(max_len):
            tgt_mask = self.generate_square_subsequent_mask(i+1).type_as(src_mask)            
            cxt = self.embed(ys) 
            # (i+1, batch, emb)
            out = self.decoder(cxt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
            # (i+1, batch, emb)

            log_probs = self.generator(out[-1:,...]) # (1, batch, vocab)
            next_words = F.gumbel_softmax(log_probs, tau=gumbel_tau, hard=False, dim=-1)

            ys = torch.cat((ys, next_words), dim=0) # (i+2, batch, vocab)
        
        if not keep_bos:
            ys = ys[1:,...]
        # (max_len+1, batch, vocab) -> (max_len, batch, vocab)
        if return_index:
            ys = ys.argmax(dim=-1)
        return ys.transpose(0, 1)

        # gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1)

