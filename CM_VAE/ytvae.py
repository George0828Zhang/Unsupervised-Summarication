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

class YT_VAE(nn.Transformer):
    def __init__(self, vocab_size, latent_size, emb_tied=False,dropout=0.5, **kwargs):
        super().__init__(dropout=dropout, **kwargs)

        self.vocab_size = vocab_size
        self.emb_tied = emb_tied
        self.latent_size = latent_size

        self.embeddings = nn.Embedding(vocab_size, self.d_model)
        self.positional = PositionalEncoding(self.d_model, dropout)

        self.hid2mean = nn.Linear(self.d_model, self.latent_size)
        self.hid2logv = nn.Linear(self.d_model, self.latent_size)
        self.lat2hid = nn.Linear(self.latent_size, self.d_model)

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
        
    def encode(self, src, src_mask):
        src_emb = self.embed(src) # (batch, xlen, emb)
        
        memory = self.encoder(src_emb.transpose(0, 1), src_key_padding_mask=src_mask)
        # (xlen, batch, d_model)
        hidden = memory[0]

        return hidden
    
    def forward(self, src, src_pad_mask, tgt, tgt_pad_mask):
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

        hidden = self.encode(src, src_pad_mask)

        mean = self.hid2mean(hidden)
        logv = self.hid2logv(hidden)
        std = torch.exp(0.5*logv)

        z = torch.randn([batch_size, self.latent_size]).type_as(hidden)
        z = z*std + mean

        kld = (-0.5 * torch.sum(logv - torch.pow(mean, 2) - torch.exp(logv) + 1, dim = 1)).mean()
        #DECODER

        de_hid = self.lat2hid(z).unsqueeze(0)
        # (1, batch, emb)
        tgt_future_mask = self.generate_square_subsequent_mask(tgt.shape[1]).type_as(src_pad_mask)
        tgt_emb = self.embed(tgt)
        # (batch, y_len, emb_size)
        output = self.decoder(tgt_emb.transpose(0, 1), de_hid, tgt_mask = tgt_future_mask, 
            tgt_key_padding_mask = tgt_pad_mask)
        logp = self.generator(output)

        return logp, kld
