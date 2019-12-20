import torch
from torch.utils import data
import numpy as np
import json
import random
import logging
from tqdm.auto import tqdm, trange

class Dataset(data.Dataset):    
    def __init__(self, src_list, nxt_list, SRC_MAX, NXT_MAX, pad_idx, cutoff=None, group_by_nxt=False):                
        if cutoff is not None:
            src_list = src_list[:cutoff]
            nxt_list = nxt_list[:cutoff]

        assert len(src_list) == len(nxt_list)

        self.size = len(src_list)
            
        self.src = []
        self.nxt = []
        
        self.pad_idx = pad_idx
        
        for i in trange(self.size):#tqdm(range(self.size)):
            src = src_list[i][:SRC_MAX]
            nxt = nxt_list[i][:NXT_MAX]
            self.src.append(src)
            self.nxt.append(nxt)
        
        len_dist = [len(x) for x in self.nxt] if group_by_nxt else [len(x) for x in self.src]
        idx = np.argsort(len_dist)[::-1] # descending
        
        self.src = [ self.src[i] for i in idx]
        self.nxt = [ self.nxt[i] for i in idx]
        
      
    def np_jagged(self, array):
        MAX = max([len(i) for i in array])
        out = [ a + [self.pad_idx]*(MAX-len(a)) if len(a) < MAX else a[:MAX] for a in array ]
        return np.asarray(out, dtype=np.int64)

    def at(self, i, batch_size=1):
        fr = i*batch_size
        to = min(fr+batch_size, self.size)
        src = self.np_jagged(self.src[fr:to])
        nxt = self.np_jagged(self.nxt[fr:to])
        return torch.from_numpy(src), torch.from_numpy(nxt)

    def __len__(self):
        return self.size

class Loader(object):
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # preprocess
        total = dataset.size // batch_size
        if total * batch_size < dataset.size:
            total += 1
        
        self.total = total
                    
    def __iter__(self):
        if self.shuffle:            
            r = list(range(self.total))
            random.shuffle(r)
            self.iters = iter(r)
        else:
            self.iters = iter(range(self.total))
        return self
    
    def __next__(self):
        return self.next()
    
    def next(self):
        index = next(self.iters)
        return self.dataset.at(index, self.batch_size)