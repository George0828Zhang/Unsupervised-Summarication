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
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer

from sklearn.metrics import f1_score

import wandb

def _one_hot(y, n_dims):
    y_hot = torch.zeros(*y.shape, n_dims).view(-1, n_dims).type_as(y)
    y_hot.scatter_(1, y.view(-1, 1), 1)
    y_hot = y_hot.view(*y.shape, -1)
    return y_hot

def tensor_replace(x, fr, to):
    out = x.clone()
    out[x==fr] = to
    return out

class Solver:
    def __init__(self, args, tokenizer, 
        data_generator,
        summarizer=None, 
        discriminator=None, 
        optimizer_G=None, 
        optimizer_D=None ):  
        # Params
        self.use_wandb = args.use_wandb
        self.batch_size = args.batch_size
        self.gumbel_tau = 1.
        self.summ_length = args.summ_length
        self.step = 0
        self.use_custom_loss = args.use_custom_loss

        if self.use_wandb:
            wandb_resume = False
            wandb.init(project="information-bottleneck-gumbelsoftmax", resume=wandb_resume)
            wandb.config.update({
                "batch size": args.batch_size,
                "input len":args.max_seq_length,
                "summary len":args.summ_length,       
                "weight decay": {"D":args.weight_decay_D, "G":args.weight_decay_G},
                "learning rate": {"D":args.learning_rate_D, "G":args.learning_rate_G}
                })
        
        # Others
        self.steps_per_save = args.steps_per_save
        self.device = torch.device("cpu" if args.no_cuda else "cuda")
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.out_dir = args.output_dir

        if not os.path.isdir(self.out_dir):
            logging.warning("Output directory does not exist. Creating one now...")
            os.makedirs(self.out_dir, exist_ok=True)

        # Models
        if summarizer is not None:
            logging.info("Sending summarizer to device...")
            self.summarizer = summarizer.to(self.device)
        if discriminator is not None:
            logging.info("Sending discriminator to device...")
            self.discriminator = discriminator.to(self.device)
        
        
        # Optims
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D

        # Data
        self.data_generator = data_generator

        logging.info("Solver init done.")


    def id2sent(self, ids, pad="<s>"):
        toks = self.tokenizer.decode(ids)
        return toks.replace("<|endoftext|>", pad)

    # def tstring(r):
    #     return ", ".join([format(f, ".5g") for f in r.cpu().numpy()])

    def solve(self, start, epochs, stage=1):
        logging.info("Start training from epoch {}. Total {} epochs.".format(start, epochs))
        self.step = 0
        for e in range(start, epochs+1):
            self.gumbel_tau = max(1e-3, 2**(1-start))
            bigbar = tqdm(self.data_generator, total=self.data_generator.total, 
                desc="[epoch] {}, tau=".format(e, self.gumbel_tau))
            
            for i, (src, nxt) in enumerate(bigbar):
                src = src.to(self.device)
                nxt = nxt.to(self.device)

                d_loss = self.train_D(src=src, nxt=nxt, N=1, update=True)
                g_loss, ys = self.train_G(src=src, nxt=nxt, max_len=self.summ_length, 
                    N=1, update=True, stage=stage, custom_loss=self.use_custom_loss)
                                
                ### logging
                bigbar.set_postfix(
                    d_loss=d_loss,
                    g_loss=g_loss,
                    # cov_loss=cov_loss,
                    )

                info = {
                        "input":self.id2sent(src[0].cpu().numpy()),
                        "output":self.id2sent(ys[0].cpu().numpy()),
                        "next":self.id2sent(nxt[0].cpu().numpy()),
                        "d_loss":d_loss,
                        "g_loss":g_loss,
                              }
                if self.use_wandb: 
                    wandb.log(info)
                else:
                    print(info)
                ###########                

                if i % 5000 == 4999:
                    self.checkpoint()

                self.step += 1
                if (self.step + 1) % self.steps_per_save == 0:
                    self.checkpoint(step=self.step)
                    
            self.checkpoint(epoch=e)

    def checkpoint(self, step=None, epoch=None, save_whole=False, save_discriminator=False):
        if epoch is not None:
            name = os.path.join(self.out_dir, "checkpoint-epoch{}".format(epoch))
        elif step is not None:
            name = os.path.join(self.out_dir, "checkpoint-step{}".format(step))
        else:
            name = os.path.join(self.out_dir, "checkpoint-fresh")

        logging.info("Saving {} to file: {}".format("model" if save_whole else "states", name))

        if save_discriminator:
            if save_whole:
                torch.save({"model":self.discriminator}, name)
            else:
                torch.save({"state":self.discriminator.state_dict()}, name)
        else:
            if save_whole:
                torch.save({"model":self.summarizer}, name)
            else:
                torch.save({"state":self.summarizer.state_dict()}, name)
        logging.info("Saving complete.")


    def train_G(self, src, nxt, max_len, N=1, update=True, stage=1, custom_loss=True):

        self.summarizer.train()
        self.discriminator.train()

        for _ in range(N):
            src_mask = (src != self.pad_token_id) #not used in PG
            # ys_hot, covloss = self.summarizer(src, src_mask=src_mask, max_len=max_len, 
            #         start_symbol=self.pad_token_id, gumbel_tau=self.gumbel_tau, return_index=False, keep_bos=False)

            ys_hot = self.summarizer(src, src_mask=src_mask, max_len=max_len, 
                    start_symbol=self.pad_token_id, gumbel_tau=self.gumbel_tau, return_index=False, keep_bos=False)

            if stage == 2:
                # concat with nxt
                nxt_hot = _one_hot(nxt, self.vocab_size).type_as(ys_hot)
                ys_nxt_hot = torch.cat((ys_hot, nxt_hot), dim=1)

                
                lm_input = ys_nxt_hot[:,:-1]
                xent_input = ys_nxt_hot[:,1:].contiguous()
            else:
                # use 0:-1
                lm_input = ys_hot[:,:-1]
                # use 1:end
                xent_input = ys_hot[:,1:].contiguous()

            # calculate mask
            lm_index = lm_input.argmax(dim=-1)          
            attn_mask = (lm_index != self.pad_token_id)
            ys = lm_index[:,:max_len]
            """attention_mask: (optional) torch.FloatTensor of shape (batch_size, sequence_length):
            Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
            """
            if custom_loss:
                lm_logits = self.discriminator(lm_input, attention_mask=attn_mask)
                log_p_LM = F.log_softmax(lm_logits, dim=-1)
                # gets 1:end

                b,s,v = xent_input.shape
                xent = F.kl_div(log_p_LM.view(b*s, v), xent_input.view(b*s, v), reduction="mean") # (b, s)
                xent[lm_index==self.pad_token_id] = 0
                xent = xent.mean()# reduction
            else:
                lm_index = tensor_replace(lm_index, self.pad_token_id, -1)
                xent, lm_logits = self.discriminator(lm_input, attention_mask=attn_mask, labels=lm_index)

            gloss = xent #+ covloss

            gloss /= N
            gloss.backward()

        if update:
            self.optimizer_G.step()
            self.optimizer_G.zero_grad()
            # clear gradient computed for G but accumulated on D
            # self.optimizer_D.zero_grad()
            self.discriminator.zero_grad()



        return gloss.item(), ys

    def train_D(self, src, nxt, N=1, update=True):
        
        self.discriminator.train()

        # concat with nxt
        src_nxt = torch.cat((src, nxt), dim=1)
        labels = tensor_replace(src_nxt, self.pad_token_id, -1) # the xent in gpt2 ignores -1 as label

        attn_mask = (src_nxt != self.pad_token_id)
        # real_loss = discriminator.inference(tgt[:,1:], start_index=vocab[BOS], ignore_index=vocab[PAD], return_prob=False).mean()
        for _ in range(N):
            real_loss, lm_logits = self.discriminator(src_nxt, attention_mask=attn_mask, labels=labels)
            real_loss /= N
            real_loss.backward()
         
        if update:
            self.optimizer_D.step()
            self.optimizer_D.zero_grad()
        
        return real_loss.item()


    def pretrain(self, start, epochs):
        logging.info("Start pretraining from epoch {}. Total {} epochs.".format(start, epochs))
        self.step = 0
        for e in range(start, epochs+1):
            bigbar = tqdm(self.data_generator, total=self.data_generator.total, 
                desc="[epoch] {}".format(e))
            
            for i, (src, lbl) in enumerate(bigbar):
                src = src.to(self.device)
                lbl = lbl.to(self.device)
                
                attn_mask = (src != self.pad_token_id)

                outputs = self.discriminator(src, nli_labels=lbl, attention_mask=attn_mask)
                nli_loss = outputs[0]
                nli_loss.backward()
             
                self.optimizer_D.step()
                self.optimizer_D.zero_grad()

                       
                ### logging
                bigbar.set_postfix(
                    nli_loss=nli_loss.item()
                    # cov_loss=cov_loss,
                    )

                info = {
                        "nli_loss":nli_loss.item()
                              }
                if self.use_wandb: 
                    wandb.log(info)
                else:
                    print(info)
                ###########                

                if i % 1000 == 999:
                    self.checkpoint(save_discriminator=True)

                self.step += 1
                if (self.step + 1) % self.steps_per_save == 0:
                    self.checkpoint(step=self.step, save_discriminator=True)
                    
            self.checkpoint(epoch=e, save_discriminator=True)

    def pretest(self):
        logging.info("Start evaluation for pretraining stage.")
        
        bigbar = tqdm(self.data_generator, total=self.data_generator.total, 
                desc="Evaluation")
            
        avg_loss = []
        _pred = []
        _real = []
        for i, (src, lbl) in enumerate(bigbar):
            src = src.to(self.device)
            lbl = lbl.to(self.device)
            
            attn_mask = (src != self.pad_token_id)

            outputs = self.discriminator(src, nli_labels=lbl, attention_mask=attn_mask)
            nli_loss, nli_logits = outputs[:2]

            _pred.append(nli_logits.cpu().detach())
            _real.append(lbl.cpu().detach())

            ### logging
            bigbar.set_postfix(
                nli_loss=nli_loss.item(),
                )

            avg_loss.append(nli_loss.item())

        all_logits = torch.cat(_pred, dim=0).view(-1, self.discriminator.n_labels)
        all_y = torch.cat(_real, dim=0).view(-1)

        f1 = self.f1_compute(logits=all_logits, y=all_y)

        print("average loss:", np.mean(avg_loss))
        print("f1:", f1)

    def f1_compute(self, logits, y):
        # everyone uses macro, but class imbalance should use micro
        batch_size = y.shape[0]
        _, predict = torch.max(logits, dim=1)
        
        y = y.type(predict.dtype)

        # to cpu
        y = y.cpu()
        predict = predict.cpu()

        f1 = f1_score(y_true=y.view(batch_size,-1), y_pred=predict.view(batch_size,-1), average='macro')

        return f1

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


class GPT2Discriminator(nn.Module):
    def __init__(self, n_labels, prototype):
        super().__init__()        
        self.transformer = prototype.transformer
        self.lm_head = prototype.lm_head
        self.n_labels = n_labels
        self.nli_head = nn.Linear(prototype.config.n_embd, n_labels)

    def forward(
        self,        
        word_ids, 
        lm_labels=None,
        nli_labels=None,
        *args, **kwargs,
    ):
        if word_ids.dim() < 2:
            raise RuntimeError("word_ids should be (batch, len) or (batch, len, vocab)")
        elif word_ids.dim() == 2:
            transformer_outputs = self.transformer(word_ids,*args, **kwargs)
        elif word_ids.dim() == 3:
            tok_emb = torch.matmul(word_ids, self.transformer.wte.weight)
            transformer_outputs = self.transformer(inputs_embeds=tok_emb,*args, **kwargs)

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if lm_labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        nli_logits = self.nli_head(hidden_states[:,-1:])
        outputs = (nli_logits,) + outputs

        if nli_labels is not None:
            nli_loss = F.cross_entropy(nli_logits.view(-1, nli_logits.size(-1)), nli_labels.view(-1))
            outputs = (nli_loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


class GPT2LM(nn.Module):
    def __init__(self, preload='distilgpt2'):
        super().__init__()
        if isinstance(preload, str):
            self.gpt2 = GPT2LMHeadModel.from_pretrained(preload)
        else:
            self.gpt2 = preload
        self.vocab_size = self.gpt2.config.vocab_size

    def forward(self, word_ids, *args, **kwargs):
        if word_ids.dim() < 2:
            raise RuntimeError("word_ids should be (batch, len) or (batch, len, vocab)")
        elif word_ids.dim() == 2:
            outputs = self.gpt2(word_ids,*args, **kwargs)
        elif word_ids.dim() == 3:
            tok_emb = torch.matmul(word_ids, self.gpt2.transformer.wte.weight)
            outputs = self.gpt2(inputs_embeds=tok_emb,*args, **kwargs)        
        # outputs = (loss,) logits, past
        if len(outputs)==2:
            return outputs[0]

        return outputs[:2]

class GPT2Summarizer(nn.Module):
    def __init__(self, preload='distilgpt2'):
        super().__init__()
        if isinstance(preload, str):
            self.gpt2 = GPT2LMHeadModel.from_pretrained(preload)
        else:
            self.gpt2 = preload
        self.vocab_size = self.gpt2.config.vocab_size

        tokenizer = GPT2Tokenizer.from_pretrained(preload)
        self.delimiter = torch.tensor(tokenizer.encode("TL;DR:")) # Size([4])

    def forward(self, src, src_mask, max_len, start_symbol, gumbel_tau=1., return_index=False, keep_bos=False):
        delim = self.delimiter.repeat((src.size(0),1)).type_as(src)
        context = torch.cat((src, delim), dim=1) # batch, len

        generated = []
        past = None

        for i in range(max_len):            
            output, past = self.gpt2(context, past=past) # batch, len, vocab
            
            token_hot = F.gumbel_softmax(output[:,-1:,:], tau=gumbel_tau, hard=False, dim=-1) # (batch, 1, vocab)
            token = token_hot.argmax(dim=-1)
            context = token
            
            generated.append(token if return_index else token_hot)
        

        ys = torch.cat(generated, dim=1)
        return ys

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
