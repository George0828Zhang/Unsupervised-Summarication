import os
import json
import numpy as np
from tqdm import tqdm#tqdm_notebook as tqdm
from multiprocessing import Pool
from subprocess import check_output
from allennlp.modules.elmo import batch_to_ids
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper

from transformers import GPT2Tokenizer

UNK = "<unk>"
BOS = ELMoCharacterMapper.bos_token
EOS = ELMoCharacterMapper.eos_token
PAD = "<pad>"

class Preprocessor(object):
    def __init__(self, doc_name, summ_name, validation_split, vocab_size, token_mappings, num_threads):
        self.summaries = []
        self.documents = []
        self.vocab_size = vocab_size
        self.token_mappings = token_mappings
        self.threads = num_threads
        self.validation_split = validation_split

        with open(doc_name, newline='', encoding='utf-8') as f:
            total = self.getlines(doc_name)
            for i,line in tqdm(enumerate(f), total=total):
                line = self.swap(line.strip())
                self.documents.append(line)
        if summ_name:
            with open(summ_name, newline='', encoding='utf-8') as f:
                total = self.getlines(summ_name)
                for i,line in tqdm(enumerate(f), total=total):
                    line = self.swap(line.strip())
                    self.summaries.append(line)
                
        self.size = len(self.documents)
        
    
    def process(self, vocab=None, lower=True):
        print("[info] making vocabulary...")
        self.make_vocab(lower=lower)
        
        if vocab is not None:
            print("[info] using external vocabulary !!!")
            self.vocab = vocab
        self.vocab_inv = {a:b for b, a in self.vocab.items()}
        print("[info] converting to indices...")
        self.convert_all_to_ids()  
                
    def make_vocab(self, lower):
        sum_toks = []
        doc_toks = []
        vocab = {}

        for d in tqdm(self.summaries):
            if lower:
                d = d.lower()
            ts = d.split()
            for t in ts:
                vocab[t] = vocab.get(t, 0) + 1
            sum_toks.append(ts)

        for d in tqdm(self.documents):
            if lower:
                d = d.lower()
            ts = d.split()
            for t in ts:
                vocab[t] = vocab.get(t, 0) + 1
            doc_toks.append(ts)
            
        vocab_sort = [(PAD, None), (BOS, None), (EOS, None)] + sorted(vocab.items(), key=lambda x: -x[1]) # descending
        print(vocab_sort[:50])
        self.vocab = { v:i for i, (v, n) in enumerate(vocab_sort[:self.vocab_size])}
        self.summaries = sum_toks
        self.documents = doc_toks
        
    def tokens_to_ids(self, s):
        return [self.vocab[BOS]] + [self.vocab.get(t, self.vocab[UNK]) for t in s] + [self.vocab[EOS]]
    
    def ids_to_tokens(self,ids):
        return [self.vocab_inv[i] for i in ids]
    
    def convert_all_to_ids(self):
        if self.threads < 2:
            self.summ_seqs = [self.tokens_to_ids(s) for s in tqdm(self.summaries)]
            self.docu_seqs = [self.tokens_to_ids(s) for s in tqdm(self.documents)]
        else:
            self.summ_seqs = [self.tokens_to_ids(s) for s in tqdm(self.summaries)]
            self.docu_seqs = [self.tokens_to_ids(s) for s in tqdm(self.documents)]
            # with Pool(self.threads) as p:
            #     self.summ_seqs = list(p.imap(self.tokens_to_ids, tqdm(self.summaries)))
            #     self.docu_seqs = list(p.imap(self.tokens_to_ids, tqdm(self.documents)))
    
    def export(self, vocab_name=None, data_seq_name="tmp.json", valid_seq_name=None):
        if vocab_name is not None:
            print("[info] dumping vocab...")
            json.dump(self.vocab, open(vocab_name, 'w'))
        
        seqdata = {'summary':[], 'document':[]}
        valseqdata = {'summary':[], 'document':[]}
        
        if self.validation_split > 0:
            print("[info] splitting data...")
            num_summ = self.size
            val_set = np.random.randint(0, num_summ, size=int(self.validation_split*num_summ))
            for i in range(num_summ):
                if i in val_set:
                    valseqdata['summary'].append(self.summ_seqs[i])
                    valseqdata['document'].append(self.docu_seqs[i])
                else:
                    seqdata['summary'].append(self.summ_seqs[i])
                    seqdata['document'].append(self.docu_seqs[i])
            
            print("[info] dumping validation data...")
            json.dump(valseqdata, open(valid_seq_name, 'w'))
        else:
            seqdata['summary'] = self.summ_seqs
            seqdata['document'] = self.docu_seqs
        
        print("[info] dumping training data...")
        json.dump(seqdata, open(data_seq_name, 'w'))
        
        
        
    def swap(self,s):
        for t, t_p in self.token_mappings.items():
            s = s.replace(t, t_p)
        if s == "":
            s = UNK
        return s
    
    def getlines(self,name):
        #total = !wc -l {name}
        #return int(total[0].split()[0])
        return int(check_output(["wc", "-l", name]).split()[0])

class GPT2Preprocessor(Preprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.EOSid = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')
    
    def process(self, vocab=None, add_EOS=False):        
        
        self.summ_seqs = [self.tokenizer.encode(s, add_prefix_space=True) for s in tqdm(self.summaries)]
        self.docu_seqs = [self.tokenizer.encode(s, add_prefix_space=True) for s in tqdm(self.documents)]
        
        if add_EOS:
            self.summ_seqs = [[self.EOSid] + s + [self.EOSid] for s in self.summ_seqs]
            self.docu_seqs = [[self.EOSid] + s + [self.EOSid] for s in self.docu_seqs]

    def export(self, vocab_name=None, data_seq_name="tmp.json", valid_seq_name=None):
        if vocab_name is not None:
            print("[info] dumping vocab...")
            self.tokenizer.save_vocabulary(os.path.dirname(vocab_name))
        
        seqdata = {'summary':[], 'document':[]}
        valseqdata = {'summary':[], 'document':[]}
        
        if self.validation_split > 0:
            print("[info] splitting data...")
            num_summ = self.size
            
            val_set = int(self.validation_split*num_summ)

            valseqdata['summary'] = self.summ_seqs[-val_set:]
            valseqdata['document'] = self.docu_seqs[-val_set:]

            seqdata['summary'] = self.summ_seqs[:-val_set]
            seqdata['document'] = self.docu_seqs[:-val_set]
            
            print("[info] dumping validation data...")
            json.dump(valseqdata, open(valid_seq_name, 'w'))
        else:
            seqdata['summary'] = self.summ_seqs
            seqdata['document'] = self.docu_seqs
        
        print("[info] dumping training data...")
        json.dump(seqdata, open(data_seq_name, 'w'))




def main():
    task_name = "giga"
    task_type = "train"
    out_dir = "/hdd/data-giga-gpt2-withEOF/"#"data-{}/".format(task_name)
    num_threads = 4
    validation_split = 0.005 if task_type == "train" else 0
    add_EOS = True

    if task_name == "giga":
        doc_name = "/hdd/giga/train.article.txt"
        summ_name = "/hdd/giga/train.title.txt"
        if task_type == "eval":
            doc_name = "/home/george/Projects/Datasets/giga/test.article.txt"
            summ_name = "/home/george/Projects/Datasets/giga/test.title.txt"
    elif task_name == "wiki103":
        doc_name = "/tmp2/b05902064/wikitext-103/wiki.train.tokens.2"
        summ_name = ""
        if task_type == "eval":
            doc_name = "/tmp2/b05902064/wikitext-103/wiki.valid.tokens.2"
        validation_split = 0
    else:
        doc_name = "../pointer-generator/data2/train.txt.src"
        summ_name = "../pointer-generator/data2/train.txt.tgt.tagged"

    
    vocab_name = out_dir+"vocab.json"
    data_seq_name = out_dir+ ("train_seq.json" if task_type == "train" else "test_seq.json")
    valid_seq_name = out_dir+"valid_seq.json"

    

    if task_name == 'giga':
        token_mappings = {'<unk>':UNK}#, '-lrb-':'(', '-rrb-':')'}
    else:
        token_mappings = {}    


    p = GPT2Preprocessor(doc_name, summ_name, validation_split, 20000, token_mappings, num_threads)
    if task_type == "eval" and not isinstance(p, GPT2Preprocessor):
        vocab = json.load(open(vocab_name, "r"))
        p.process(vocab=vocab, add_EOS = add_EOS)
    else:
        p.process(add_EOS = add_EOS)

    os.makedirs(out_dir,exist_ok=True)
    p.export(vocab_name,data_seq_name,valid_seq_name)

if __name__ == "__main__":
    main()

    # data = json.load(open("data-wiki103/train_seq.json", 'r'))
    # print("load json done.")
    # sum_list = data['summary']
    # data_list = data['document']

    # print(len(sum_list),len(data_list))

    # import matplotlib.pyplot as plt
    # lens = [len(s) for s in data_list]
    # plt.hist(lens, bins=50)
    # plt.show()
    # print(np.mean(lens), np.std(lens))
