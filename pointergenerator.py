import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerGenerator(nn.Module):
    def __init__(self, vocab_size, d_model, d_emb, nhead, num_layers=2, dropout=0.5, coverage=True, eps=1e-10):
        super().__init__()
        
        self.d_model = d_model
        self.d_emb = d_emb
        self.vocab_size = vocab_size
        self.epsilon = eps
        self.coverage = coverage
        self.num_layers = num_layers
        
        if coverage:
            self.cov_weight = nn.Parameter(torch.randn(1, dtype=torch.float)/10)

        self.embeddings = nn.Embedding(vocab_size, d_emb)
        self.positional = nn.Dropout(dropout)

        self.encoder = nn.LSTM(d_emb, d_model, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(d_emb, d_model*2, num_layers=num_layers, batch_first=True)
                
        self.pro_layer = nn.Sequential(
            nn.Linear(d_model*4, vocab_size, bias=True),
            nn.Softmax(dim=-1)
        )
        self.pgen_layer = nn.Sequential(
            nn.Linear(4*d_model+d_emb, 1, bias=True),
            nn.Sigmoid()
        )

    def embed(self, x):
        if x.dim() < 2:
            raise RuntimeError("x should be (batch, len) or (batch, len, vocab)")
        elif x.dim() == 2:
            emb = self.embeddings(x)
        elif x.dim() == 3:
            emb = torch.matmul(x, self.embeddings.weight)
        return self.positional(emb)

        
    def forward(self, src, src_mask, max_len, start_symbol, gumbel_tau=1., return_index=False, keep_bos=False):
        batch_size, xlen = src.shape[:2]

        # encoder
        src_emb = self.embed(src) # (batch, xlen, emb)        
        memory, (h, c) = self.encoder(src_emb)
        # (batch, xlen, emb),  (num_layers * 2, batch, hidden_size)

        # unpack direction
        h = h.view(self.num_layers, 2, batch_size, self.d_model)
        c = c.view(self.num_layers, 2, batch_size, self.d_model)

        # transform to decoder (L, 2, b, h) -> (L, b, 2*h)
        out_h = h.transpose(1, 2).contiguous().view(self.num_layers, batch_size, 2*self.d_model)
        out_c = c.transpose(1, 2).contiguous().view(self.num_layers, batch_size, 2*self.d_model)
        
        # use gumbel softmax instead        
        # one-hot
        ys = torch.zeros((1, batch_size, self.vocab_size)).type_as(memory)
        ys[...,start_symbol] = 1.
        
        if self.coverage:
            covloss = torch.zeros(1).type_as(memory)
            coverage = [torch.zeros([batch_size,1, xlen]).type_as(memory)]
        
        for i in range(max_len):
            # 3 dimensional
            ans_emb = self.embed(ys[-1:,...].transpose(0,1).detach()) #(batch, 1, emb)
            out, (out_h, out_c) = self.decoder(ans_emb, (out_h, out_c)) #(batch, 1, 2hidden)
            
            ######
            if self.coverage:
                attention = torch.matmul(out, memory.transpose(-1, -2)) #(batch, 1, srclen)
                attention = attention + coverage[-1] * self.cov_weight
                attention = F.softmax(attention, dim=-1)
                covloss += torch.min(attention, coverage[-1]).sum() / batch_size
                coverage.append(coverage[-1]+attention)
            else:
                attention = torch.matmul(out, memory.transpose(-1, -2)) #(batch, 1, srclen)
                attention = F.softmax(attention, dim=-1)
            ######
            
            context_vector = torch.matmul(attention, memory) #(batch, 1, 2hidden)

            pointer_prob = torch.zeros((batch_size, self.vocab_size)).type_as(attention)
            pointer_prob = pointer_prob.scatter_add_(dim=1, index=src, src=attention.squeeze())
            
            feature = torch.cat((out, context_vector), -1) #(batch, 1, 4hidden)
            pgen_feat = torch.cat((feature, ans_emb), -1) #(batch, 1, 4hidden+emb)
            
            ### 3 dimensional to 2 dimensional
            distri = self.pro_layer(feature.squeeze()) #(batch, vocab)
            pgen = self.pgen_layer(pgen_feat.squeeze()) #(batch, 1)
            
            assert (pgen >= 0).all()
            assert (distri >= 0).all()

            final_dis = pgen*distri + (1.-pgen)*pointer_prob + self.epsilon
            assert (final_dis > 0).all()
            
            log_probs = final_dis.log().unsqueeze(0) #(1, batch, vocab)

            next_words = F.gumbel_softmax(log_probs, tau=gumbel_tau, hard=False, dim=-1)

            ys = torch.cat((ys, next_words), dim=0) # (i+2, batch, vocab)
        
        if not keep_bos:
            ys = ys[1:,...]
        # (max_len+1, batch, vocab) -> (max_len, batch, vocab)
        if return_index:
            ys = ys.argmax(dim=-1)        
        if self.coverage:
            return ys.transpose(0, 1), covloss
        return ys.transpose(0, 1)
           