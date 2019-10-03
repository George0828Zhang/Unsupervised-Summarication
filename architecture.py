from transformer_nb2 import *

from preprocessors import BOS, EOS, PAD, UNK

from ELMo import LanguageModel, getELMo

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
    def __init__(self, vocab, lmpath):
        super().__init__()
        self.LM = LanguageModel(vocab)
        tmp = torch.load(lmpath)['model']
        self.LM.load_state_dict(tmp)
        self.pretrained_elmo = getELMo(vocab)

        self.eval()

    def elmo_embed(self, t):
        return self.pretrained_elmo(t)

    def language_model(self, t):
        return self.LM.inference(t)

    def compute_scores(self, x, y, lbd=0.11):
        # y: (batch, len)
        seqlen = y.shape[1]

        x_reps = self.elmo_embed(x) # (batch, xlen, emb)
        y_reps = self.elmo_embed(y) # (batch, ylen, emb)

        # contextual matching
        scores_cm = torch.matmul(y_reps, x_reps[:,-1,:].unsqueeze(-1)).squeeze() # (batch, ylen, emb) x (batch, emb, 1) 
        # no need to normalize because all y map to x[-1]
        # is softmax needed though? probably not because this is not meant to be a distribution
        
        # domain fluency
        scores_fm = self.language_model(y)

        reward = scores_cm * scores_fm ** lbd
        return reward



def loss_compute(seq2seq, matcher, src, max_len, vocab, gamma=0.99):
    batch_size = src.shape[0]
    src_mask = (src != vocab[PAD]).unsqueeze(-2)
    ys, log_p = seq2seq(src, src_mask, max_len, vocab[BOS])

    rewards = matcher.compute_scores(src, ys, lbd=0.11) # should have same shape as ys
    
    rewards_adjust = []
    littleR = torch.zeros(batch_size)
    for t in reversed(range(max_len)):
        r = rewards[:,t]

        littleR = r + gamma*littleR
        rewards_adjust.append(littleR)    

    rewards_adjust = torch.stack(rewards_adjust[::-1], 1)

    rewardTensor =  torch.FloatTensor(rewards_adjust).type_as(log_p)


    r_mean = rewardTensor.mean(-1, keepdim=True)
    r_std = rewardTensor.std(-1, keepdim=True)

    rewardTensor = (rewardTensor - r_mean)/r_std

    losses = -torch.sum(torch.mul(log_p, rewardTensor), -1)
    # (batch)
    return losses.mean()




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

#     src = torch.ones((2, 5)).long()
#     max_len = 3
    

#     data_dir = "/home/george/Projects/Summarization-Lab/contextual-matching/data-giga/"
#     train_path = data_dir + "train_seq.json"
#     vocab_path = data_dir + "vocab.json"
#     model_path = "trained/DomainFluency"

#     vocab = json.load()
#     matcher = ContextMatcher(vocab, model_path)


#     loss = loss_compute(translator, src, max_len, vocab, gamma=0.99)

# if __name__ == "__main__":
#     main()