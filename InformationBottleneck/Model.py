import os
import math
import numpy as np
import json
import random
import logging
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel



def _one_hot(y, n_dims):
    y_hot = torch.zeros(*y.shape, n_dims).view(-1, n_dims).type_as(y)
    y_hot.scatter_(1, y.view(-1, 1), 1)
    y_hot = y_hot.view(*y.shape, -1)
    return y_hot

class Solver:
    def __init__(self, args, tokenizer, translator, discriminator, dataset, data_generator,
        optim=torch.optim.RMSprop, 
        ):  
        # Params
        self.use_wandb = args.use_wandb
        self.batch_size = args.batch_size
        self.gumbel_tau = 1.
        self.summ_len = args.summ_length
        
        # Others
        self.device = torch.device("cpu" if args.no_cuda else "cuda")
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.PAD_id = tokenizer.PAD_id
        self.out_dir = args.output_dir

        if not os.path.isdir(self.out_dir):
            logging.warning("Output directory does not exist. Creating one now...")
            os.makedirs(self.out_dir, exist_ok=True)

        # Models
        logging.info("Sending translator to device...")
        self.translator = translator.to(self.device)
        logging.info("Sending discriminator to device...")
        self.discriminator = discriminator.to(self.device)
        logging.info("Done.")
        
        # Optims
        self.optimizer_G = optim(translator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                       
        # Data
        self.data_generator = data_generator
        self.total_train = int(math.ceil(dataset.size / self.batch_size))


    def id2sent(self, ids):
        toks = self.tokenizer.decode(ids)
        return toks.replace("<|endoftext|>", "")

    # def tstring(r):
    #     return ", ".join([format(f, ".5g") for f in r.cpu().numpy()])

    def solve(self, start, epochs):
        logging.info("Start training from epoch {}. Total {} epochs.".format(start, epochs))

        if self.use_wandb:
            import wandb
            wandb_resume = False
            wandb.init(project="information-bottleneck-gumbelsoftmax", resume=wandb_resume)
            wandb.config.update({
                "batch size": self.batch_size,
                "input len":self.max_seq_length,
                "summary len":self.summ_length,       
                "weight decay": args.weight_decay,
                "learning rate": args.learning_rate
                })


        for e in range(start, epochs+1):
            self.gumbel_tau = max(1e-3, 2**(1-start))
            bigbar = tqdm(self.data_generator, total=self.total_train, 
                desc="[epoch] {}, tau=".format(e, self.gumbel_tau))
            
            for i, (src, nxt) in enumerate(bigbar):
                src = src.to(self.device)
                nxt = nxt.to(self.device)

                g_loss, cov_loss, ys = self.train_G(src=src, nxt=nxt, max_len=self.summ_len, N=1, update=True)
                                
                ### logging
                bigbar.set_postfix(
                    g_loss=g_loss,
                    # cov_loss=cov_loss,
                    )

                info = {
                        "input":self.id2sent(src[0].cpu().numpy()),
                        "output":self.id2sent(ys[0].cpu().numpy()),
                        "next":self.id2sent(nxt[0].cpu().numpy()),
                        "g_loss":g_loss,
                        "cov loss":cov_loss,
                              }
                if self.use_wandb: 
                    wandb.log(info)
                else:
                    print(info)
                ###########                

                if i % 5000 == 4999:
                    self.checkpoint()
                    
            self.checkpoint(epoch=e)

    def checkpoint(self, step=None, epoch=None, save_whole=False):
        if epoch is not None:
            name = os.path.join(self.out_dir, "checkpoint-epoch{}".format(epoch))
        elif step is not None:
            name = os.path.join(self.out_dir, "checkpoint-step{}".format(step))
        else:
            name = os.path.join(self.out_dir, "checkpoint-fresh")

        logging.info("Saving {} to file: {}".format("model" if save_whole else "states", name))
        if save_whole:
            torch.save({"model":self.translator}, name)
        else:
            torch.save({"state":self.translator.state_dict()}, name)
        logging.info("Saving complete.")


    def train_G(self, src, nxt, max_len, N=1, update=True):

        self.translator.train()
        self.discriminator.train()

        for _ in range(N):
            src_mask = (src == self.PAD_id) 
            ys_hot, covloss = self.translator(src, src_mask=src_mask, max_len=max_len, 
                    start_symbol=self.PAD_id, gumbel_tau=self.gumbel_tau, return_index=False, keep_bos=True)


            # concat with nxt
            nxt_hot = _one_hot(nxt, self.vocab_size).type_as(ys_hot)
            ys_nxt_hot = torch.cat((ys_hot, nxt_hot), dim=1)
            
            # use 0:-1
            lm_input = ys_nxt_hot[:,:-1] #.argmax(dim=-1)
            lm_logits = self.discriminator(lm_input)
            log_p_LM = F.log_softmax(lm_logits, dim=-1)
            # gets 1:end
            

            # use 1:end
            xent_input = ys_nxt_hot[:,1:].contiguous()
            b,s,v = xent_input.shape
            xent = F.kl_div(log_p_LM.view(b*s, v), xent_input.view(b*s, v), reduction="batchmean") # (b, s)

            gloss = xent + covloss

            gloss /= N
            gloss.backward()
            # (batch, len)
            #gloss = (xent.mean() + covloss)/N
            #gloss.backward()

        if update:
            self.optimizer_G.step()
            self.optimizer_G.zero_grad()
            if hasattr(self, "optimizer_D"):
                self.optimizer_D.zero_grad() # clear gradient computed for G

        return gloss.item(), covloss.item(), ys_hot.argmax(dim=-1)



class PointerGenerator(nn.Module):
    def __init__(self, vocab_size, d_model, d_emb, num_layers=2, dropout=0.5, coverage=True, eps=1e-10):
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


class GPT2LM(nn.Module):
    def __init__(self, preload='distilgpt2'):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(preload)
        self.vocab_size = self.gpt2.config.vocab_size

    def forward(self, word_ids):
        if word_ids.dim() < 2:
            raise RuntimeError("word_ids should be (batch, len) or (batch, len, vocab)")
        elif word_ids.dim() == 2:
            logits, past = self.gpt2(word_ids)
        elif word_ids.dim() == 3:
            tok_emb = torch.matmul(word_ids, self.gpt2.transformer.wte.weight)
            logits, past = self.gpt2(inputs_embeds=tok_emb)        

        return logits

    # def inference(self, sent, start_index, ignore_index=-100, return_prob=False):
    #     # (batch, len)
    #     batch_size, seqlen = sent.shape[:2]
    #     src = torch.ones(batch_size, 1).fill_(start_index).type_as(sent.data)
    #     src = torch.cat((src, sent[:,:-1]), 1)
    #     tgt = sent.contiguous()
      
    #     logits = self.forward(src) # (1, len, vocab)
                   
    #     CE = F.cross_entropy(logits.view(-1, self.vocab_size), tgt.view(-1), 
    #         ignore_index=ignore_index, reduction='none').view(batch_size, seqlen)

    #     if not return_prob:
    #         return CE
    #     else:
    #         return (-CE).exp()




# from pytransformer import PositionalEncoding

# class LanguageModel(nn.Module):
#     def __init__(self, vocab, emb_dim, hidden_dim, dropout, emb_share=True, use_position=True):
#         super().__init__()        
#         self.vocab = vocab
#         self.vocab_size = len(vocab)
#         self.emb_share = emb_share

#         self.embed = nn.Embedding(self.vocab_size, emb_dim)
#         self.position = PositionalEncoding(emb_dim, dropout) if use_position else nn.Dropout(dropout)
#         self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=2, dropout=dropout, batch_first=True, bidirectional=False)
        
#         if not emb_share:
#             self._project = nn.Linear(hidden_dim, self.vocab_size) 

#     def project(self, h):
#         # (batch, len, hidden)
#         if self.emb_share:
#             proj = F.linear(h, self.embed.weight)
#         else:
#             proj = self._project(h)
#         return proj
   
#     def decode(self, src, max_len, mode='sample'):
#         batch_size = src.size(0)
#         word_ids = src[:,:1] # should be BOS (batch, 1)
#         logits = []

#         for i in range(max_len):
#             emb = self.position(self.embed(word_ids[:,-1:])) # (batch, 1, emb)
#             out, (h, c) = self.lstm(emb, None if i == 0 else (h, c)) # (batch, 1, hidden)
#             proj = self.project(out)
#             proj = F.log_softmax(proj, dim=-1) # (batch, 1, vocab)

#             if mode == 'argmax':
#                 values, next_words = torch.max(proj, dim=-1, keepdim=True)
#             elif mode == 'sample':
#                 m = torch.distributions.Categorical(logits=proj)
#                 next_words = m.sample()
#             else:
#                 raise

#             logits.append(proj)
#             word_ids = torch.cat((word_ids, next_words), dim=1)
#         logits = torch.cat(logits, dim=1)
#         return word_ids[:,1:], logits


#     def forward(self, word_ids):
#         if word_ids.dim() < 2:
#             raise RuntimeError("word_ids should be (batch, len) or (batch, len, vocab)")
#         elif word_ids.dim() == 2:
#             emb = self.position(self.embed(word_ids))
#         elif word_ids.dim() == 3:
#             emb = torch.matmul(word_ids, self.embed.weight)
#             emb = self.position(emb)
#         out, (h, c) = self.lstm(emb)

#         proj = self.project(out)
#         return proj #F.log_softmax(proj, dim=-1)
    
#     def inference(self, sent, start_index, ignore_index=-100, return_prob=False):
#         # (batch, len)
#         batch_size, seqlen = sent.shape[:2]
#         src = torch.ones(batch_size, 1).fill_(start_index).type_as(sent.data)
#         src = torch.cat((src, sent[:,:-1]), 1)
#         tgt = sent.contiguous()
        
#         logits = self.forward(src) # (1, len, vocab)
            
#         CE = F.cross_entropy(logits.view(-1, self.vocab_size), tgt.view(-1), 
#             ignore_index=ignore_index, reduction='none').view(batch_size, seqlen)

#         if not return_prob:
#             return CE
#         else:
#             return (-CE).exp()
