import os
import csv
import torch
from torch.utils import data
import numpy as np
import json
import random
import logging
from tqdm.auto import tqdm, trange


from transformers import GPT2Tokenizer

class BasicProcessor:
    def load(self, path):
        indices = torch.load(path)
        # quick fix
        if "tgt" in indices and "nxt" not in indices:
            indices["nxt"] = indices["tgt"]

        self.prepro_result = indices

    def save(self, path):
        logging.warning("Caching results to file: "+path)
        torch.save(self.prepro_result, path)
        logging.warning("Done!")

    def process(self,):
        raise NotImplementedError

    def make_dataset(self):
        raise NotImplementedError
      
    def np_jagged(self, array):
        MAX = max([len(i) for i in array])
        out = [ a + [self.pad_idx]*(MAX-len(a)) if len(a) < MAX else a[:MAX] for a in array ]
        return np.asarray(out, dtype=np.int64)

    def at(self, i, batch_size=1):
        raise NotImplementedError

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

    def __len__(self):
        return self.total
                    
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

class NextSentenceProcessor(BasicProcessor):
    def __init__(self, dirpath, tokenizer, phase="train", min_bytes=75):        
        
        self.min_bytes = min_bytes
        self.tokenizer = tokenizer

        self.path = os.path.join(dirpath, "{}.src.txt".format(phase))
    
    def process(self, cutoff=None):
        logging.info(f"Reading data from file: {self.path}")
        with open(self.path, "r") as f:
            data = f.readlines()
        
        sentences, candidates = self.example_pairs(data[:cutoff])
        total_sents = len(sentences)
        logging.info("Total extracted sentences: {}".format(total_sents))
        filtered = {}
        for c in candidates:
            filtered[c] = sentences[c]
            filtered[c+1] = sentences[c+1]        

        logging.info("Total remaining sentences: {}".format(len(filtered)))
        sequences = self.encode_all(filtered)
             
        src = []
        nxt = []
        for i in candidates:
            src.append(sequences[i])
            nxt.append(sequences[i+1])

        assert len(src) == len(nxt)
        assert None not in src
        assert None not in nxt
        
        self.prepro_result = {"src":src, "nxt":nxt}
        logging.info("Complete. Total examples: {}".format(len(src)))
        return self.prepro_result
            
    
    def example_pairs(self, data):
        logging.info("Extracting sentence pairs...")

        sents_pool = []
        candidates = []
        base = 0
        for d in tqdm(data):
            sents = nltk.sent_tokenize(d)            
            sents_pool.extend(sents)
            N = len(sents)
            
            for i in range(N-1):
                if len(sents[i]) >= self.min_bytes and len(sents[i+1]) >= self.min_bytes:
                    candidates.append(base + i)
            base += N

        logging.info("Done.")
        return sents_pool, candidates
                
    def encode_all(self, data):
        logging.info("Encoding sentences into indices...")        
        seqs = {}
        for i,s in tqdm(data.items()):
            seqs[i] = self.tokenizer.encode(s, add_prefix_space=True)
        logging.info("Done.")
        return seqs

    ####################
    # make dataset now #
    ####################

    def make_dataset(self, SRC_MAX, NXT_MAX, pad_idx, cutoff=None, group_by_nxt=False):                
        if cutoff is not None:
            src_list = self.prepro_result["src"][:cutoff]
            nxt_list = self.prepro_result["nxt"][:cutoff]

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

    def at(self, i, batch_size=1):
        fr = i*batch_size
        to = min(fr+batch_size, self.size)
        src = self.np_jagged(self.src[fr:to])
        nxt = self.np_jagged(self.nxt[fr:to])
        return torch.from_numpy(src), torch.from_numpy(nxt)

class GigawordProcessor(BasicProcessor):
    def __init__(self, dirpath, tokenizer, phase="train"):        
        self.tokenizer = tokenizer

        self.src_path = os.path.join(dirpath, "{}.src.txt".format(phase))
        self.tgt_path = os.path.join(dirpath, "{}.tgt.txt".format(phase))
    
    def process(self, cutoff=None):
        logging.info(f"Reading data from file: {self.src_path}")
        with open(self.src_path, "r") as f:
            src_data = f.readlines()
        logging.info(f"Reading data from file: {self.tgt_path}")
        with open(self.tgt_path, "r") as f:
            tgt_data = f.readlines()
        
        assert len(src_data) == len(tgt_data)

        src_data = {i:x for i,x in enumerate(src_data[:cutoff])}
        tgt_data = {i:x for i,x in enumerate(tgt_data[:cutoff])}

        total_sents = len(src_data)
        logging.info("Total extracted sentences: {}".format(total_sents))

        src_seqs = self.encode_all(src_data)
        tgt_seqs = self.encode_all(tgt_data)
             
        src = []
        tgt = []
        for i in range(total_sents):
            src.append(src_seqs[i])
            tgt.append(tgt_seqs[i])

        assert len(src) == len(tgt)
        assert None not in src
        assert None not in tgt
        
        self.prepro_result = {"src":src, "tgt":tgt}
        logging.info("Complete. Total examples: {}".format(len(src)))
        return self.prepro_result
                            
    def encode_all(self, data):
        logging.info("Encoding sentences into indices...")        
        seqs = {}
        for i,s in tqdm(data.items()):
            seqs[i] = self.tokenizer.encode(s, add_prefix_space=True)
        logging.info("Done.")
        return seqs

    ####################
    # make dataset now #
    ####################

    def make_dataset(self, SRC_MAX, NXT_MAX, pad_idx, cutoff=None, group_by_nxt=False):                
        if cutoff is not None:
            src_list = self.prepro_result["src"][:cutoff]
            nxt_list = self.prepro_result["tgt"][:cutoff]

        assert len(src_list) == len(nxt_list)

        self.size = len(src_list)
            
        self.src_tgt = []
        
        self.pad_idx = pad_idx

        BOS = [self.tokenizer.bos_token_id]
        CLS = [self.tokenizer.cls_token_id]
        SEP = [self.tokenizer.sep_token_id]
        for i in trange(self.size):
            src = src_list[i][:SRC_MAX]
            tgt = nxt_list[i][:NXT_MAX]
            fmt = BOS + src + SEP + tgt + CLS
            self.src_tgt.append(fmt)
        
        self.src_tgt = sorted(self.src_tgt, key=lambda x: -len(x)) # descending

        # len_dist = [len(x) for x in self.src_tgt]
        # idx = np.argsort(len_dist)[::-1] # descending
        
        # self.src_tgt = [ self.src[i] for i in idx]
        # self.nxt = [ self.nxt[i] for i in idx]

    def at(self, i, batch_size=1):
        fr = i*batch_size
        to = min(fr+batch_size, self.size)
        src_tgt = self.np_jagged(self.src_tgt[fr:to])
        return torch.from_numpy(src_tgt)

class SNLIProcessor(BasicProcessor):
    def __init__(self, dirpath, tokenizer, phase="train"):        
        self.tokenizer = tokenizer

        self.src_path = os.path.join(dirpath, "snli_1.0_{}.txt".format(phase))
    
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def process(self, cutoff=None):
        logging.info(f"Reading data from file: {self.src_path}")

        src_data = []
        tgt_data = []
        lbl_data = []

        """
        gold_label   sentence1_binary_parse  sentence2_binary_parse  sentence1_parse sentence2_parse sentence1   sentence2   captionID   pairID  label1  label2  label3  label4  label5"""
        
        label_map = {"neutral": 0, "entailment":1, "contradiction":2}
        lines = self._read_tsv(self.src_path)
        for fields in lines[1:]:            
            src_data.append(fields[5])
            tgt_data.append(fields[6])
            
            if fields[0] == "-":
                # 9 - 13
                vote = [0,0,0]
                for j in fields[9:]:
                    if j in label_map:
                        vote[label_map[j]] += 1
                lbl = np.argmax(vote)
            else:
                lbl = label_map[fields[0]]

            lbl_data.append(lbl)
        
        assert len(src_data) == len(tgt_data)
        assert len(src_data) == len(lbl_data)

        src_data = {i:x for i,x in enumerate(src_data[:cutoff])}
        tgt_data = {i:x for i,x in enumerate(tgt_data[:cutoff])}

        total_sents = len(src_data)
        logging.info("Total extracted sentences: {}".format(total_sents))

        src_seqs = self.encode_all(src_data)
        tgt_seqs = self.encode_all(tgt_data)
             
        src = []
        tgt = []
        for i in range(total_sents):
            src.append(src_seqs[i])
            tgt.append(tgt_seqs[i])

        assert len(src) == len(tgt)
        assert None not in src
        assert None not in tgt
        
        self.prepro_result = {"src":src, "tgt":tgt, "lbl":lbl_data}
        logging.info("Complete. Total examples: {}".format(len(src)))
        return self.prepro_result
                            
    def encode_all(self, data):
        logging.info("Encoding sentences into indices...")        
        seqs = {}
        for i,s in tqdm(data.items()):
            seqs[i] = self.tokenizer.encode(s, add_prefix_space=True)
        logging.info("Done.")
        return seqs

    ####################
    # make dataset now #
    ####################

    def make_dataset(self, SRC_MAX, NXT_MAX, pad_idx, cutoff=None, group_by_nxt=False):                
        if cutoff is not None:
            src_list = self.prepro_result["src"][:cutoff]
            tgt_list = self.prepro_result["tgt"][:cutoff]
            lbl_list = self.prepro_result["lbl"][:cutoff]

        assert len(src_list) == len(tgt_list)
        assert len(src_list) == len(lbl_list)
       
            
        self.src_tgt = []
        self.labels = []
        
        self.pad_idx = pad_idx

        BOS = [self.tokenizer.bos_token_id]
        CLS = [self.tokenizer.cls_token_id]
        SEP = [self.tokenizer.sep_token_id]
        for i in trange(len(src_list)):
            src = src_list[i][:SRC_MAX]
            tgt = tgt_list[i][:NXT_MAX]
            self.src_tgt.append(BOS + src + SEP + tgt + CLS)
            self.src_tgt.append(BOS + tgt + SEP + src + CLS) # data augmentation
            self.labels.append(lbl_list[i])
            self.labels.append(lbl_list[i]) # data augmentation
        
        # self.src_tgt = sorted(self.src_tgt, key=lambda x: -len(x)) # descending

        self.size = len(self.src_tgt)

        len_dist = [len(x) for x in self.src_tgt]
        idx = np.argsort(len_dist)[::-1] # descending
        
        self.src_tgt = [ self.src_tgt[i] for i in idx]
        self.labels = [ self.labels[i] for i in idx]

    def at(self, i, batch_size=1):
        fr = i*batch_size
        to = min(fr+batch_size, self.size)
        src_tgt = self.np_jagged(self.src_tgt[fr:to])
        padded = torch.from_numpy(src_tgt)
        return padded, torch.LongTensor(self.labels[fr:to])
        # clstail = torch.tensor([self.tokenizer.cls_token_id]).repeat(padded.size(0), 1)
        # return torch.cat((padded, clstail), dim=1), torch.LongTensor(self.labels[fr:to])