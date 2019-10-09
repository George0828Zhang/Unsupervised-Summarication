from transformer_nb2 import *

from preprocessors import BOS, EOS, PAD, UNK

from ELMo import LanguageModel, getELMo
import torch.nn.functional as F
from tqdm.auto import tqdm, trange

def freeze(m: nn.Module):
    for p in list(m.parameters()):
        p.requires_grad = False

class Translator(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Translator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, src_mask, max_len, start_symbol):
        "Take in and process masked src and target sequences."
        
        batch_size = src.shape[0]

        memory = self.encode(src, src_mask)
        ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src.data)
        logits = []
        
        for i in range(max_len):
            out = self.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src_mask))
            log_probs = self.generator(out[:, -1, :])
            M = torch.distributions.Categorical(logits=log_probs)
            next_words = M.sample()

            ys = torch.cat((ys, next_words.unsqueeze(1)), dim=1)
            
            # values, _ = torch.max(log_probs, dim=-1, keepdim=True)
            logits.append(M.log_prob(next_words))
        
        logits = torch.stack(logits,1)
        return ys[:,1:], logits
            
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class ContextMatcher(nn.Module):
    def __init__(self, embeddings, elmo, LM, candidate_map=None):
        """
        embeddings: pretrained embeddings, should have shape (vocab, emb_dim)
        elmo: pretrained elmo module
        LM: finetuned elmo+linear module 
        """
        super().__init__()        
        self.LM = LM        
        self.pretrained_elmo = elmo
        self.embeddings = embeddings
        
        self.eps = 1e-9

        if not isinstance(self.embeddings, torch.Tensor):
            self.embeddings = torch.Tensor(self.embeddings)

        # l2-norm
        self.embeddings = F.normalize(self.embeddings, p=2, dim=-1)

        freeze(self.LM)

        if candidate_map is None:
            self.candidate_map = self._get_candidate_mapping()
        else:
            self.candidate_map = candidate_map
        
    def _get_candidate_mapping(self, batch_size = 128):
        vocab_size, emb_dim = self.embeddings.shape
                
        total = int(math.ceil(vocab_size/batch_size))
        
        output = []
        for i in trange(total, desc="candidate-mappings"):
            # (50000, 1024)
            fr = i*batch_size
            to = min(fr+batch_size, vocab_size)
            cur = self.embeddings[fr:to] #(batch, 1024)
            scores = torch.matmul(cur, self.embeddings.transpose(0, 1)) #(batch, 50000)
            candidates = torch.argsort(scores, dim=-1, descending=True) #(batch, 50000)
            output.append(candidates[:,:50].cpu()) #(batch, 50) # send to cpu to free memory
        return torch.cat(output, dim=0)
        
    
    def candidate_list(self, x, K):  
        assert K <= 50
        assert x.dim() <= 1
        pool = self.candidate_map[x,:K] # (xlen, K)
        return torch.unique(pool)
       
    def voronoi_split(self, candidates):
        assert candidates.dim() == 1
        cand_emb = self.embeddings[candidates,:] #(C, emb)
        scores = torch.matmul(self.embeddings, cand_emb.transpose(0,1)) #(V, C)
        cell_num = torch.argmax(scores, dim=1).squeeze() #(V,)
        self.voronoi_mapping = cell_num

    def get_cell_mates(self, wid):
        if not hasattr(self, "voronoi_mapping"):
            print("execute voronoi_split(candidates) first.")
            return 
        cell = self.voronoi_mapping[wid]
        return (self.voronoi_mapping == cell).nonzero()

    def _runtest(self, vocab, sent, qword):
        vocab_inv = {a:b for b,a in vocab.items()}
        words = sent.lower().split()
        
        wids = [vocab.get(word, vocab[UNK]) for word in words]
        
        print("got:", [vocab_inv[w] for w in wids], "wids:", wids)
        
        wids = torch.LongTensor(wids)
        
        out = self.candidate_list(wids, K=6)
        candi = out
        print("candidates:", [vocab_inv[i.item()] for i in candi])
               
        ### voronoi
        self.voronoi_split(candi)
        
        ### testing
        word = qword.lower().strip()
        
        wid = vocab.get(word, vocab[UNK])
        
        print("got:", vocab_inv[wid], "wids:", wid)
        
        mates = self.get_cell_mates(wid)
        print("cell mates:", [vocab_inv[i.item()] for i in mates])

    def embed(self, t):
        dummy = torch.zeros((t.shape[0], t.shape[1], 50)).type_as(t)        
        embeddings = self.pretrained_elmo(dummy, word_inputs=t)
        return embeddings['elmo_representations'][0]

    def language_model(self, t):
#         return torch.log(self.LM.inference(t)+self.eps)
        return self.LM.inference(t)

    

    def contextual_matching(self, x, y):
        # y: (batch, len)
        seqlen = y.shape[1]

        x_reps = self.embed(x) # (batch, xlen, emb)
        y_reps = self.embed(y) # (batch, ylen, emb)

        # l2-norm on last embedding        
        y_reps = F.normalize(y_reps, p=2, dim=-1) # (batch, ylen, emb)

        # should try max and mean
        x_reps = x_reps.mean(dim=1, keepdim=True) # (batch, 1, emb)
        x_reps = F.normalize(x_reps, p=2, dim=-1) # (batch, 1, emb)

        cosine = torch.matmul(y_reps, x_reps.transpose(-2, -1)) # (batch, ylen, 1)
        #scores, _ = torch.max(cosine, dim=2) # (batch, ylen)
        # likelihood = torch.sigmoid(cosine.squeeze())
        return cosine.squeeze()

    def compute_scores(self, x, y, lbd=0.11):
        # contextual matching score
        scores_cm = self.contextual_matching(x, y)
        
        # domain fluency score
        scores_fm = self.language_model(y)
        scores_fm = 2*scores_fm - 1 # according to language GAN paper

        reward = scores_cm + scores_fm * lbd
        return reward, (scores_cm, scores_fm)



def rewards_compute(matcher, src, ys, log_p, adjust, zero_mean, unit_standard, baseline=0, gamma=0.99, eps=1e-9):
    batch_size = src.shape[0]
    max_len = ys.shape[1]
    
    rewards, (cm, fm) = matcher.compute_scores(src, ys, lbd=1) # should have same shape as ys
    
    # should we adjust the rewards?
    if adjust:
        rewards_adjust = []
        littleR = torch.zeros(batch_size).type_as(rewards)
        for t in reversed(range(max_len)):
            r = rewards[:,t]

            littleR = r + gamma*littleR
            rewards_adjust.append(littleR)    

        rewardTensor = torch.stack(rewards_adjust[::-1], 1)
    else:
        rewardTensor = rewards

    # should we standarize the rewards?
    
    # r_mean = rewardTensor.mean()#-1, keepdim=True)
    # r_std = rewardTensor.std()#-1, keepdim=True)
    # if zero_mean:
    #     rewardTensor = rewardTensor - r_mean
    # if unit_standard:
    #     rewardTensor = rewardTensor/(r_std+eps)
    dis = 0.99
    baseline = baseline*dis + rewardTensor.mean().item()*(1-dis)
    rewardTensor = rewardTensor - baseline


    final_reward = torch.sum(torch.mul(log_p, rewardTensor), -1)

    # (batch)
    return final_reward, (cm, fm), baseline




def make_translator(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1, emb_share=False):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    src_emb = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    tgt_emb = src_emb if emb_share else nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
    
    model = Translator(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        src_emb,
        tgt_emb,
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class PointerGenerator(nn.Module):
    def __init__(self, hidden_dim, emb_dim, input_len, output_len, voc_size, coverage=True, eps=1e-8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.input_len = input_len
        self.output_len = output_len
        self.voc_size = voc_size
        self.teacher_prob = 1.
        self.epsilon = eps
        self.coverage = coverage
        
        if coverage:
            self.cov_weight = nn.Parameter(torch.randn(1, dtype=torch.float)/10)

        self.emb_layer = nn.Embedding(voc_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim*2, num_layers=1, batch_first=True)
                
        self.pro_layer = nn.Sequential(
            nn.Linear(hidden_dim*4, voc_size, bias=True),
            nn.Softmax(dim=-1)
        )
        self.pgen_layer = nn.Sequential(
            nn.Linear(4*hidden_dim+emb_dim, 1, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, src, src_mask, max_len, start_symbol, mode = 'sample'):
        x = src
        batch_size, xlen = x.shape[:2]
        device = x.device
        
        # encoder
        x_emb = self.emb_layer(x)
        memory, (h, c) = self.encoder(x_emb) #(batch, srclen, 2hidden)
        h = h.transpose(0, 1).contiguous()
        c = c.transpose(0, 1).contiguous()
        h = h.view(batch_size, 1, h.shape[-1]*2)
        c = c.view(batch_size, 1, c.shape[-1]*2)
        h = h.transpose(0, 1).contiguous()
        c = c.transpose(0, 1).contiguous()        

        
        ## decoder
        out_h, out_c = (h, c)        
        
        ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(x.data)
        
        log_probs_seq = []

        if self.coverage:
            covloss = torch.zeros(1).type_as(memory)
            coverage = [torch.zeros([batch_size,1, xlen]).type_as(memory)]
        
        for i in range(max_len):
            # 3 dimensional
            ans_emb = self.emb_layer(ys[:,-1].unsqueeze(1)) #(batch, 1, emb)
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

            attention = torch.matmul(out, memory.transpose(-1, -2)) #(batch, 1, srclen)
            attention = F.softmax(attention, dim=-1)
            
            context_vector = torch.matmul(attention, memory) #(batch, 1, 2hidden)

            pointer_prob = torch.zeros((batch_size, self.voc_size)).type_as(attention)
            pointer_prob = pointer_prob.scatter_add_(dim=1, index=x, src=attention.squeeze())
            
            feature = torch.cat((out, context_vector), -1) #(batch, 1, 4hidden)
            pgen_feat = torch.cat((context_vector, out, ans_emb), -1) #(batch, 1, 4hidden+emb)
            
            ### 3 dimensional to 2 dimensional
            distri = self.pro_layer(feature.squeeze()) #(batch, vocab)
            pgen = self.pgen_layer(pgen_feat.squeeze()) #(batch, 1)
            
            assert (pgen >= 0).all()
            assert (distri >= 0).all()

            final_dis = pgen*distri + (1.-pgen)*pointer_prob + self.epsilon
            assert (final_dis > 0).all()
            
            log_probs = final_dis.log() #(batch, 1, vocab)
                
            if mode == 'argmax':
                values, next_words = torch.max(log_probs, dim=-1, keepdim=True)
            if mode == 'sample':
                m = torch.distributions.Categorical(logits=log_probs)
                next_words = m.sample()
                values = m.log_prob(next_words)
                
            # all_log_probs.append(log_probs)    
            ys = torch.cat((ys, next_words.unsqueeze(1)), dim=1)
            
            log_probs_seq.append(values)
        
        log_probs_seq = torch.stack(log_probs_seq,1)
        if self.coverage:
            return ys[:,1:], log_probs_seq, covloss
        return ys[:,1:], log_probs_seq

    def forward_teacher(self, src, src_mask, tgt, max_len, start_symbol, mode = 'sample'):
        x = src
        batch_size = x.shape[0]
        device = x.device
        
        # encoder
        x_emb = self.emb_layer(x)
        memory, (h, c) = self.encoder(x_emb) #(batch, srclen, 2hidden)
        h = h.transpose(0, 1).contiguous()
        c = c.transpose(0, 1).contiguous()
        h = h.view(batch_size, 1, h.shape[-1]*2)
        c = c.view(batch_size, 1, c.shape[-1]*2)
        h = h.transpose(0, 1).contiguous()
        c = c.transpose(0, 1).contiguous()        

        
        ## decoder
        out_h, out_c = (h, c)        
        
        ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(x.data)
        
        logits = []
        tgtlen = tgt.shape[1]
        for i in range(max_len):
            # 3 dimensional
            cand = ys[:,-1:]#tgt[:,i:i+1] if i<tgtlen else ys[:,-1:]
            
            ans_emb = self.emb_layer(cand) #(batch, 1, emb)
            out, (out_h, out_c) = self.decoder(ans_emb, (out_h, out_c)) #(batch, 1, 2hidden)
            
            attention = torch.matmul(out, memory.transpose(-1, -2)) #(batch, 1, srclen)
            attention = F.softmax(attention, dim=-1)
            
            context_vector = torch.matmul(attention, memory) #(batch, 1, 2hidden)
            
            pointer_prob = torch.zeros((batch_size, self.voc_size)).type_as(attention)
            pointer_prob = pointer_prob.scatter_add_(dim=1, index=x, src=attention.squeeze())
            
            feature = torch.cat((out, context_vector), -1) #(batch, 1, 4hidden)
            pgen_feat = torch.cat((context_vector, out, ans_emb), -1) #(batch, 1, 4hidden+emb)
            
            ### 3 dimensional to 2 dimensional
            distri = self.pro_layer(feature.squeeze()) #(batch, vocab)
            pgen = self.pgen_layer(pgen_feat.squeeze()) #(batch, 1)
            
            assert (pgen >= 0).all()
            assert (distri >= 0).all()

            final_dis = pgen*distri + (1.-pgen)*pointer_prob + self.epsilon
            assert (final_dis > 0).all()
            
            log_probs = final_dis.log() #(batch, 1, vocab)
                
            if mode == 'argmax':
                values, next_words = torch.max(log_probs, dim=-1, keepdim=True)
            if mode == 'sample':
                m = torch.distributions.Categorical(logits=log_probs)
                next_words = m.sample()
                values = m.log_prob(next_words)
                
            # all_log_probs.append(log_probs)    
            ys = torch.cat((ys, next_words.unsqueeze(1)), dim=1)
            
            logits.append(log_probs)
        
        logits = torch.stack(logits,1)
        return ys[:,1:], logits
