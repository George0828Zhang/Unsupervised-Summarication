import numpy as np
import torch
import torch.nn as nn

from allennlp.modules.elmo import Elmo, batch_to_ids
import json
import random
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper
from preprocessors import BOS, EOS, PAD, UNK # special tokens
import math
from tqdm import tqdm_notebook as tqdm

def getELMo(vocab):
    options_file = "data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    return Elmo(options_file, weight_file, num_output_representations=1, dropout=0.5, vocab_to_cache=list(vocab.keys()))

class LanguageModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        
        # options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        # weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.elmo = getELMo(vocab)
        self.project = nn.Linear(1024, self.vocab_size)
        self.CE = nn.CrossEntropyLoss(ignore_index=self.vocab[PAD], reduction='none')
    
    def forward(self, word_ids):
        dummy = torch.zeros((word_ids.shape[0], word_ids.shape[1], 50)).type_as(word_ids)
        embeddings = self.elmo(dummy, word_inputs=word_ids)
        distribution = self.project(embeddings['elmo_representations'][0])
        return distribution
              

    def inference(self, sent):
        # (batch, len)
        batch_size, seqlen = sent.shape[:2]
        src = torch.ones(batch_size, 1).fill_(self.vocab[BOS]).type_as(sent.data)
        src = torch.cat((src, sent[:,:-1]), 1)
        tgt = sent      
        
        logits = self.LM(src) # (1, len, vocab)
        CE = self.CE(logits.view(-1, self.vocab_size), tgt.view(-1))
        probs = (-CE).exp()
        return probs.view(batch_size, seqlen)

class DomainFluency(nn.Module):
    def __init__(self, modelpath, vocabpath, device='cuda'):
        super().__init__()

        self.vocab = json.load(open(vocabpath))
        self.device = torch.device(device)
        self.CE = nn.CrossEntropyLoss(ignore_index=self.vocab[PAD]).to(self.device)

        tmp = torch.load(modelpath)['model']
        self.LM = LanguageModel(len(self.vocab)).to(device)
        self.LM.load_state_dict(tmp)
        self.LM.eval()

    def p_lm(self, sent: list):
        last_id = self.vocab.get(sent[-1], self.vocab[UNK])
        
        in_sent = [BOS] + sent[:-1]
        x = batch_to_ids([in_sent])
        with torch.no_grad():
            logits = self.LM(x.to(self.device)) # (1, len, vocab)
            logits = logits[:,-1,:]
            tgt = torch.LongTensor([last_id]).to(self.device)
            CE = self.CE(logits.view(-1, len(self.vocab)), tgt.view(-1))
            prob = (-CE).exp().item()
        return prob
