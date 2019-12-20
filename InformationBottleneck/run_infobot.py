#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import logging
import torch
import nltk
import numpy
from tqdm.auto import tqdm
from multiprocessing import Pool

# _projdir = os.path.abspath(os.path.dirname(__name__))
# print(_projdir)
# sys.path.insert(0, _projdir)

# from dataset import Dataset
import dataset
from Model import Solver, GPT2LM, PointerGenerator


def main():
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] [%(funcName)s] %(message)s',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stderr)]
        )
    logging.info("Logger initialized successfully.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/george/Projects/Datasets/cnndm",
        # default=None, required=True, 
        type=str, help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task", type=str, default="cnn", 
        # default=None, required=True,
                            help="Task to run. (cnn, wikitext, etc.)")
    parser.add_argument('--overwrite_cache', action='store_true',
                            help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--use_wandb", action='store_true',
                        help="Whether to use wandb.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    # prepro args
    parser.add_argument("--cutoff", default=100000000, type=int,
                        help="Cutoff for training examples. Default 1e8.")

    # train args
    parser.add_argument("--shuffle", action='store_true',
                        help="Whether to shuffle the training examples.")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")

    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float,
    #                     help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--start_epoch", default=1, type=int,
                        help="Starting epoch.")

    parser.add_argument("--min_byte_length", default=75, type=int,
                        help="The minimum total input sequence length in bytes before tokenization."
                        "Sequences shorter will be discard.")

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--summ_length", default=20, type=int,
                        help="The maximum total generated summary sequence length.")

    parser.add_argument("--group_by_nxt", action='store_true',
                        help="Whether to group training sequence based on next sentence (nxt). Default is group by source (src).")
    args = parser.parse_args()
    
    if args.do_train:
        phase="train"
    elif args.do_eval:
        phase="val"

    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.PAD = '<|endoftext|>'
    tokenizer.PAD_id = tokenizer.convert_tokens_to_ids(tokenizer.PAD)

    data_set, data_generator = load_and_cache_examples(args, tokenizer, phase=phase)

    #### test
    # for src, nxt in data_generator:
    #     for a, b in zip(src, nxt):
    #         a = tokenizer.decode(a.numpy())
    #         b = tokenizer.decode(b.numpy())
    #         a = a.replace("<|endoftext|>", "")
    #         b = b.replace("<|endoftext|>", "")
    #         print("{}||{}".format(a, b))
    #         input("next?")
    #########
    
    discriminator = GPT2LM(preload='distilgpt2')
    translator = PointerGenerator(vocab_size=discriminator.vocab_size, 
        d_model=256, d_emb=256, num_layers=2, dropout=0.1, coverage=True)
    

    solver = Solver(args, tokenizer=tokenizer, 
        translator=translator, 
        discriminator=discriminator, 
        dataset=data_set, data_generator=data_generator,
        optim=torch.optim.RMSprop,
        )
    solver.solve(start=args.start_epoch, epochs=args.num_train_epochs)


def load_and_cache_examples(args, tokenizer, phase="train"):    
    outname = "{}.{}.bin".format(args.task, phase)
    outpath = os.path.join(args.data_dir, outname)
    if not os.path.isfile(outpath) or args.overwrite_cache:
        logging.warning("No cached file, or overwrite specified. Running preprocessing.")
        doc_name, sum_name = None, None
        for fname in os.listdir(args.data_dir):
            if "txt" in fname and phase in fname:
                if "src" in fname:
                    logging.info("Found document file: "+fname)
                    doc_name = os.path.join(args.data_dir, fname)
                elif "nxt" in fname:
                    logging.info("Found summary file: "+fname)
                    sum_name = os.path.join(args.data_dir, fname)
        if doc_name is not None:
            p = GPT2Preprocessor(
                path=doc_name, 
                min_bytes=args.min_byte_length,
                tokenizer=tokenizer
                )
            indices = p.process(cutoff=args.cutoff)
            logging.warning("Caching results to file: "+outpath)
            torch.save(indices, outpath)
            logging.warning("Done!")
    else:
        logging.info("Loading detected cache file: "+outpath)
        indices = torch.load(outpath)

    if "tgt" in indices and "nxt" not in indices:
        indices["nxt"] = indices["tgt"]

    logging.info("Batching training examples...")
    dset = dataset.Dataset(indices["src"], indices["nxt"], 
        SRC_MAX=args.max_seq_length, 
        NXT_MAX=args.max_seq_length, 
        pad_idx=tokenizer.PAD_id, 
        cutoff=args.cutoff, 
        group_by_nxt=args.group_by_nxt)
    loader = dataset.Loader(dset, batch_size=args.batch_size, shuffle=args.shuffle)
    
    return dset, loader



from transformers import GPT2Tokenizer

class GPT2Preprocessor:
    def __init__(self, path, min_bytes=75, tokenizer=None):
        logging.info("Initializing GPT2Preprocessor...")
        self.path = path
        self.min_bytes = min_bytes
        self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2') if tokenizer==None else tokenizer        
        logging.info("Initialized successfully.")

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
        
        self.result = {"src":src, "nxt":nxt}
        logging.info("Complete. Total examples: {}".format(len(src)))
        return self.result
            
    
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

if __name__ == "__main__":
    main()