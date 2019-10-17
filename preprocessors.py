import json
import numpy as np
from tqdm import tqdm#tqdm_notebook as tqdm
from multiprocessing import Pool
from subprocess import check_output
from allennlp.modules.elmo import batch_to_ids
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper

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
        with open(summ_name, newline='', encoding='utf-8') as f:
            total = self.getlines(summ_name)
            for i,line in tqdm(enumerate(f), total=total):
                line = self.swap(line.strip())
                self.summaries.append(line)
                
        self.size = len(self.summaries)
        
    
    def process(self, vocab=None):
        print("[info] making vocabulary...")
        self.make_vocab()
        
        if vocab is not None:
            print("[info] using external vocabulary !!!")
            self.vocab = vocab
        self.vocab_inv = {a:b for b, a in self.vocab.items()}
        print("[info] converting to indices...")
        self.convert_all_to_ids()  
                
    def make_vocab(self):
        sum_toks = []
        doc_toks = []
        vocab = {}

        for d in tqdm(self.summaries):
            ts = d.split()
            for t in ts:
                vocab[t] = vocab.get(t, 0) + 1
            sum_toks.append(ts)

        for d in tqdm(self.documents):
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



def main():
    task_name = "giga"
    task_type = "train"

    if task_name == "giga":
        doc_name = "/hdd/giga/train.article.txt"
        summ_name = "/hdd/giga/train.title.txt"
        if task_type == "eval":
            doc_name = "../speechlab/pointer-generator/data/Giga/input.txt"
            summ_name = "../speechlab/pointer-generator/data/Giga/task1_ref0.txt"
    else:
        doc_name = "../pointer-generator/data2/train.txt.src"
        summ_name = "../pointer-generator/data2/train.txt.tgt.tagged"

    out_dir = "/hdd/giga/data-20k/"#"data-{}/".format(task_name)
    vocab_name = out_dir+"vocab.json"
    data_seq_name = out_dir+ ("train_seq.json" if task_type == "train" else "test_seq.json")
    valid_seq_name = out_dir+"valid_seq.json"

    num_threads = 4
    corpus_size = 888888
    validation_split = 0.005 if task_type == "train" else 0

    if task_name == 'giga':
        token_mappings = {'<unk>':UNK}#, '-lrb-':'(', '-rrb-':')'}
    else:
        token_mappings = {}
    


    p = Preprocessor(doc_name, summ_name, validation_split, 20000, token_mappings, num_threads)
    if task_type == "eval":
        vocab = json.load(open(vocab_name, "r"))
        p.process(vocab=vocab)
    else:
        p.process()
    p.export(vocab_name,data_seq_name,valid_seq_name)

if __name__ == "__main__":
    main()