from NeuralLM import LanguageModel
from dataset import *
from preprocessors import BOS, EOS, UNK, PAD
import torch
import json
import math
import torch.nn.functional as F

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
            #xentloss = lm.CE(logits.contiguous().view(-1, lm.vocab_size), tgt.contiguous().view(-1))# negative log probs           
            xentloss = F.cross_entropy(logits.contiguous().view(-1, lm.vocab_size), tgt.contiguous().view(-1), reduction='none')
            trange.set_postfix(xent=xentloss.mean().item())
            total_xent.append(xentloss) # (batch * seqlen,)
        ppl = torch.cat(total_xent, dim=0).mean().exp()
    return ppl.item()

def LM_perplex():
    batch_size = 256
    data_dir = "data-fixed/"
    preload = "GANLM/LM-check"
    vocab = json.load(open(data_dir+"vocab.json", "r"))
    vocab_size = len(vocab)

    device = torch.device("cuda")
    # model = torch.load(preload).to(device)
    model = torch.load(preload, map_location=lambda s, l: s).to(device)
    
    pool = ["test", "valid"]
    for name in pool:
        testing_set = PretrainDataset(data_dir+name+"_seq.json", 20, 20, vocab[EOS]) #train_seq

        testing_generator = Loader(testing_set, batch_size=batch_size, shuffle=False)
        total_test = int(math.ceil(testing_set.size / batch_size))

        print(name+" perplexity:", perplexity(model, testing_generator, total_test, device))

if __name__ == "__main__":
    LM_perplex()
