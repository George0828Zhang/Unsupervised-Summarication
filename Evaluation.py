from ELMo import LanguageModel
from dataset import *
from preprocessors import BOS, EOS, UNK, PAD
import torch
import json
import math

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

def perplexity(lm, corpus_gen, total, device=torch.device("cuda")):
    lm.eval()
    with torch.no_grad():  
        total_xent = []
        trange = tqdm(corpus_gen, total=total)
        # CE = nn.CrossEntropyLoss(reduction='none').to(device)
        for src, tgt in trange:
            src = src.to(device)
            tgt = tgt.to(device)
            batch_size, seqlen = tgt.shape[:2]

            logits = lm.forward(src)
            xentloss = lm.CE(logits.view(-1, lm.vocab_size), tgt.view(-1))# negative log probs            
            trange.set_postfix(xent=xentloss.mean().item())
            total_xent.append(xentloss) # (batch * seqlen,)
        ppl = torch.cat(total_xent, dim=0).mean().exp()
    return ppl.item()

def LM_perplex():
    batch_size = 256
    data_dir = "data-20k/"
    preload = "trainedELMo/LM20"
    vocab = json.load(open(data_dir+"vocab.json", "r"))
    vocab_size = len(vocab)

    device = torch.device("cuda")
    model = torch.load(preload).to(device)
    
    pool = ["test", "valid", "train"]
    for name in pool:
        testing_set = PretrainDataset(data_dir+name+"_seq.json", 20, 20, vocab[EOS]) #train_seq

        testing_generator = Loader(testing_set, batch_size=batch_size, shuffle=False)
        total_test = int(math.ceil(testing_set.size / batch_size))

        print(name+" perplexity:", perplexity(model, testing_generator, total_test))

if __name__ == "__main__":
    LM_perplex()