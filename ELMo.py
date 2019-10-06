import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.elmo import Elmo, batch_to_ids
import json
import random
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper
from preprocessors import BOS, EOS, PAD, UNK # special tokens
import math
from tqdm import tqdm_notebook as tqdm





def getELMo(vocab, unidir, downstream=False):
    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    
    vocab_to_cache=sorted(vocab.keys(), key=lambda t: vocab[t])
    if downstream:
        elmo = Elmo(options_file, weight_file, num_output_representations=1, vocab_to_cache=vocab_to_cache)
    else:
        elmo = Elmo(options_file, weight_file, num_output_representations=1, scalar_mix_parameters=[1,1,1], vocab_to_cache=vocab_to_cache)
        

    if unidir:
        for l in ["backward_layer_0", "backward_layer_1"]:
            layer = getattr(elmo._elmo_lstm._elmo_lstm, l)
            for s in ["input_linearity", "state_linearity", "state_projection"]:
                subject = getattr(layer, s)
                for a in ["weight", "bias"]:
                    if hasattr(subject, a) and getattr(subject, a) is not None:
                        target = getattr(subject, a)
                        target.data.fill_(0.0)

    return elmo 

class LanguageModel(nn.Module):
    def __init__(self, vocab, elmo):
        super().__init__()        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.elmo = elmo
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
        tgt = sent.contiguous()
        
        logits = self.forward(src) # (1, len, vocab)
        
        CE = self.CE(logits.view(-1, self.vocab_size), tgt.view(-1))
        probs = (-CE).exp()
        return probs.view(batch_size, seqlen)