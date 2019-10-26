import torch
from torch.utils import data
import numpy as np
import json
import random
from tqdm.auto import tqdm, trange

class Dataset(data.Dataset):    
    def __init__(self, data_name, INPUT_MAX, OUTPUT_MAX, pad_idx, cutoff=None):
        print("loading json")
        data = json.load(open(data_name, 'r'))
        print("load json done.")
        sum_list = data['summary']
        data_list = data['document']
        
        if cutoff is not None:
            sum_list = sum_list[:cutoff]
            data_list = data_list[:cutoff]
        # idata -> list
        self.size = len(data_list)
            
        self.src = []
        self.tgt = []
        
        self.pad_idx = pad_idx
        
        for i in trange(self.size):#tqdm(range(self.size)):
            src = data_list[i][:INPUT_MAX]
            tgt = sum_list[i][:OUTPUT_MAX]
            self.src.append(src)
            self.tgt.append(tgt)
                    
        idx = np.argsort([len(x) for x in self.src])[::-1] # descending
        
        self.src = [ self.src[i] for i in idx]
        self.tgt = [ self.tgt[i] for i in idx]
        
      
    def np_jagged(self, array):
        MAX = max([len(i) for i in array])
        out = [ a + [self.pad_idx]*(MAX-len(a)) if len(a) < MAX else a[:MAX] for a in array ]
        return np.asarray(out, dtype=np.int64)

    def at(self, i, batch_size=1):
        fr = i*batch_size
        to = min(fr+batch_size, self.size)
        src = self.np_jagged(self.src[fr:to])
        tgt = self.np_jagged(self.tgt[fr:to])
        return torch.from_numpy(src), torch.from_numpy(tgt)

    def __len__(self):
        return self.size

from collections import deque
class PretrainDataset(data.Dataset):    
    def __init__(self, data_name, OUTPUT_MIN, OUTPUT_MAX, pad_idx, cutoff=None):
        print("loading json")
        data = json.load(open(data_name, 'r'))
        print("load json done.")
        
        self.src = []
        self.size = len(data['document']) if cutoff==None else cutoff

        for d in tqdm(data['document'][:cutoff], total=self.size):
            # if OUTPUT_MIN <= len(d) <= OUTPUT_MAX:
            #     self.src.append(d)            
            chunks = len(d) // (OUTPUT_MAX+1) + 1
            for i in range(chunks):
                fr = i*(OUTPUT_MAX+1)
                to = (i+1)*(OUTPUT_MAX+1)
                if OUTPUT_MIN <= len(d[fr:to]):
                    self.src.append(d[fr:to])

        self.size = len(self.src)
        
        self.pad_idx = pad_idx
                            
        idx = np.argsort([len(x) for x in self.src])[::-1] # descending
        
        self.src = [ self.src[i] for i in idx]
        
      
    def np_jagged(self, array):
        MAX = max([len(i) for i in array])
        out = [ a + [self.pad_idx]*(MAX-len(a)) if len(a) < MAX else a[:MAX] for a in array ]
        return np.asarray(out, dtype=np.int64)

    def at(self, i, batch_size=1):
        fr = i*batch_size
        to = min(fr+batch_size, self.size)
        src = self.np_jagged(self.src[fr:to]) # (batch, len)        
        return torch.from_numpy(src)

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
        
    
    
def make_data_generator(data_name, in_max, out_max, padding_idx, batch_size, pretrain=False, cutoff=None, shuffle=True, num_workers=4):    
    if pretrain:
        data_set = PretrainDataset(data_name, in_max, out_max, padding_idx, cutoff)
    else:
        data_set = Dataset(data_name, in_max, out_max, padding_idx, cutoff)
    return data_set, Loader(data_set, batch_size, shuffle)
    #params = {'batch_size':batch_size,
    #     'shuffle': shuffle,
    #     'num_workers': num_workers}
    #generator = data.DataLoader(data_set, **params)
    #return data_set, generator