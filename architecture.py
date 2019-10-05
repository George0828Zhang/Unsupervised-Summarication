from transformer_nb2 import *

from preprocessors import BOS, EOS, PAD, UNK

from ELMo import LanguageModel, getELMo
import torch.nn.functional as F
from itertools import chain

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
    def __init__(self, vocab, lmpath, unidir):
        super().__init__()
        self.LM = LanguageModel(vocab, unidir)
        tmp = torch.load(lmpath)['model']
        self.LM.load_state_dict(tmp)
        self.pretrained_elmo = getELMo(vocab, unidir)
        self.eps = 1e-9
        
        for p in list(self.LM.parameters()) + list(self.pretrained_elmo.parameters()):
            p.requires_grad = False

    def elmo_embed(self, t):
        dummy = torch.zeros((t.shape[0], t.shape[1], 50)).type_as(t)        
        embeddings = self.pretrained_elmo(dummy, word_inputs=t)
        return embeddings['elmo_representations'][0]

    def language_model(self, t):
        return self.LM.inference(t)

    def compute_scores(self, x, y, lbd=0.11):
        # y: (batch, len)
        seqlen = y.shape[1]

        x_reps = self.elmo_embed(x) # (batch, xlen, emb)
        y_reps = self.elmo_embed(y) # (batch, ylen, emb)

        # l2-norm on embedding dim
        x_reps = F.normalize(x_reps, p=2, dim=2) 
        y_reps = F.normalize(y_reps, p=2, dim=2) 
        

        # contextual matching
        scores_cm = torch.matmul(y_reps, x_reps[:,-1,:].unsqueeze(-1)).squeeze() # (batch, ylen, emb) x (batch, emb, 1)        
        # is softmax needed though? probably not because this is not meant to be a distribution
        
        # domain fluency
        scores_fm = self.language_model(y)

        reward = (scores_cm+self.eps) * ((scores_fm+self.eps) ** lbd)
        return reward, (scores_cm, scores_fm)

    def compute_scores_fast(self, x, y):
        # y: (batch, len)
        seqlen = y.shape[1]

        x_reps = self.elmo_embed(x) # (batch, xlen, emb)
        y_reps = self.elmo_embed(y) # (batch, ylen, emb)
        y_reps = F.normalize(y_reps, p=2, dim=2) # l2-norm on embedding dim

        # contextual matching
        scores_cm = torch.matmul(y_reps, x_reps[:,-1,:].unsqueeze(-1)).squeeze() # (batch, ylen, emb) x (batch, emb, 1) 
        # no need to normalize because all y map to x[-1]
        # is softmax needed though? probably not because this is not meant to be a distribution
        
        return scores_cm


def generate(seq2seq, src, max_len, vocab):
    src_mask = (src != vocab[PAD]).unsqueeze(-2)

    ys, log_p = seq2seq(src, src_mask, max_len, vocab[BOS])

    return ys, log_p

def rewards_compute(matcher, src, ys, log_p, gamma=0.99, eps=1e-9):
    batch_size = src.shape[0]
    max_len = ys.shape[1]
    
    rewards, (cm, fm) = matcher.compute_scores(src, ys) # should have same shape as ys
    

    rewards_adjust = []
    littleR = torch.zeros(batch_size).type_as(rewards)
    for t in reversed(range(max_len)):
        r = rewards[:,t]

        littleR = r + gamma*littleR
        rewards_adjust.append(littleR)    

    rewardTensor = torch.stack(rewards_adjust[::-1], 1)

    r_mean = rewardTensor.mean(-1, keepdim=True)
    r_std = rewardTensor.std(-1, keepdim=True)

    rewardTensor = (rewardTensor - r_mean)/(r_std+eps)

    final_reward = torch.sum(torch.mul(log_p, rewardTensor), -1)

    # (batch)
    return final_reward, (cm.sum(-1), fm.sum(-1))




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


# def main():
#     translator = make_model(99, 99, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, emb_share=True)

#     src = torch.ones((64,28)).long()
#     max_len = 20
    

#     data_dir = "/home/george/Projects/Summarization-Lab/contextual-matching/data-giga/"
#     train_path = data_dir + "train_seq.json"
#     vocab_path = data_dir + "vocab.json"
#     model_path = "trainedELMo/Model5"
#     import json
#     vocab = json.load(open(vocab_path))
#     matcher = ContextMatcher(vocab, model_path, unidir=False)


#     loss = loss_compute(translator, matcher, src, max_len, vocab, gamma=0.99)

# if __name__ == "__main__":
#     main()