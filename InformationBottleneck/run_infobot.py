#!/usr/bin/env python
# coding: utf-8

import re
import os
import sys
import argparse
import logging
import torch
import nltk
import numpy
from tqdm.auto import tqdm
from multiprocessing import Pool

from dataset import *
from Model import *


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
    parser.add_argument("--task", type=str, required=True, #default="cnn", 
        # default=None, required=True,
                            help="Task to run. (cnn, wikitext, etc.)")
    parser.add_argument('--overwrite_cache', action='store_true',
                            help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_pretrain", action='store_true',
                        help="Whether to run pretraining.")
    parser.add_argument("--do_pretest", action='store_true',
                        help="Whether to run testing for pretrain stage.")
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
    parser.add_argument("--min_byte_length", default=75, type=int,
                        help="The minimum total input sequence length in bytes before tokenization."
                        "Sequences shorter will be discard.")

    # train args
    parser.add_argument("--shuffle", action='store_true',
                        help="Whether to shuffle the training examples.")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")

    parser.add_argument("--learning_rate_D", default=5e-5, type=float,
                        help="The initial learning rate for Optimizer D.")
    parser.add_argument("--weight_decay_D", default=0.0, type=float,
                        help="Weight deay if we apply some. Optimizer D.")
    parser.add_argument("--learning_rate_G", default=5e-5, type=float,
                        help="The initial learning rate for Optimizer G.")
    parser.add_argument("--weight_decay_G", default=0.0, type=float,
                        help="Weight deay if we apply some. Optimizer G.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float,
    #                     help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--start_epoch", default=1, type=int,
                        help="Starting epoch.")
    parser.add_argument("--steps_per_save", default=20000, type=int,
                        help="How often to save a checkpoint. Default is 20000 steps")
    parser.add_argument("--stage", default=1, type=int,
                        help="Starting stage. Stage 1 uses GPT2 to smoothen seq2seq output,"
                        " while stage 2 uses next sentence to enforce information bottleneck.")
    parser.add_argument("--pretrained_D", default="", type=str,
                        help="Path to pretrained discriminator model weights.")
    
    

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--summ_length", default=20, type=int,
                        help="The maximum total generated summary sequence length.")

    # model specific
    parser.add_argument("--group_by_nxt", action='store_true',
                        help="Whether to group training sequence based on next sentence (nxt). Default is group by source (src).")
    parser.add_argument("--use_custom_loss", action='store_true',
                        help="Whether to use the KL_div loss we proposed, instead of cross entropy.")
    args = parser.parse_args()
    
    


    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    # print(tokenizer.all_special_tokens)
    # print(tokenizer.all_special_ids)
    # input()
    tokenizer.add_special_tokens({
        "cls_token":"[CLS]",
        "sep_token":"[SEP]",
        "pad_token":"[PAD]",
        "bos_token":"[BOS]",
        })
    #tokenizer.vocab_size = len(tokenizer)



    #### test
    # for src, nxt in data_generator:
    #     for a, b in zip(src, nxt):
    #         a = tokenizer.decode(a.numpy())
    #         # b = tokenizer.decode(b.numpy())
    #         # a = a.replace("<|endoftext|>", "")
    #         # b = b.replace("<|endoftext|>", "")
    #         print("{}||{}".format(a, b))
    #         input("next?")
    #########

    if args.do_pretrain:
        
        prototype = GPT2LMHeadModel.from_pretrained('distilgpt2')
        prototype.resize_token_embeddings(len(tokenizer))

        discriminator = GPT2Discriminator(n_labels=3, prototype=prototype)

        if os.path.isfile(args.pretrained_D):
            tmp = torch.load(args.pretrained_D)['state']
            # tmp = { x:tmp[x] for x in ('nli_head.weight', 'nli_head.bias')}            
            discriminator.load_state_dict(tmp, strict=False)

        data_generator = load_and_cache_examples(args, tokenizer, phase="pretrain")
        

        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=args.learning_rate_D, weight_decay=args.weight_decay_D)
        
        solver = Solver(args, tokenizer=tokenizer,             
            data_generator=data_generator,
            discriminator=discriminator, 
            optimizer_D=optimizer_D,
        )
        solver.pretrain(start=args.start_epoch, epochs=args.num_train_epochs)

    elif args.do_pretest:       

        prototype = GPT2LMHeadModel.from_pretrained('distilgpt2')
        prototype.resize_token_embeddings(len(tokenizer))

        discriminator = GPT2Discriminator(n_labels=3, prototype=prototype)

        tmp = torch.load(args.pretrained_D)['state']
        # tmp = { x:tmp[x] for x in ('nli_head.weight', 'nli_head.bias')}        
        discriminator.load_state_dict(tmp, strict=False)

        data_generator = load_and_cache_examples(args, tokenizer, phase="pretest")

        solver = Solver(args, tokenizer=tokenizer, 
            data_generator=data_generator,
            discriminator=discriminator,
        )
        solver.pretest()

    elif args.do_train or args.do_eval:

        if args.do_train:
            phase="train"
        else:
            phase="val"

        prototype = GPT2LMHeadModel.from_pretrained('distilgpt2')
        prototype.resize_token_embeddings(len(tokenizer))

        discriminator = GPT2Discriminator(n_labels=3, prototype=prototype)
        tmp = torch.load(args.pretrained_D)['state']
        tmp = { x:tmp[x] for x in ('nli_head.weight', 'nli_head.bias')}
        
        discriminator.load_state_dict(tmp, strict=False)

        data_generator = load_and_cache_examples(args, tokenizer, phase="pretrain")



        raise NotImplementedError("yet to fix optimizer")
        
        data_generator = load_and_cache_examples(args, tokenizer, phase=phase)
        discriminator = GPT2LM(preload='distilgpt2')
        # summarizer = PointerGenerator(vocab_size=discriminator.vocab_size, 
        #     d_model=256, d_emb=256, num_layers=2, dropout=0.1, coverage=True)
        summarizer = GPT2Summarizer(preload='distilgpt2')
        

        solver = Solver(args, tokenizer=tokenizer, 
            summarizer=summarizer, 
            discriminator=discriminator, 
            data_generator=data_generator,
            optim=torch.optim.RMSprop,
            )
        solver.solve(start=args.start_epoch, epochs=args.num_train_epochs, stage=args.stage)


def load_and_cache_examples(args, tokenizer, phase="train"):    
    outname = "{}.{}.bin".format(args.task, phase)
    outpath = os.path.join(args.data_dir, outname)
    
    if re.match("^pre", phase) is not None:
        phase = phase.replace("pre","")
        processor = SNLIProcessor(args.data_dir, tokenizer, phase=phase)
    else:
        processor = NextSentenceProcessor(args.data_dir, tokenizer, phase=phase, min_bytes=args.min_byte_length)

    if not os.path.isfile(outpath) or args.overwrite_cache:
        logging.warning("No cached file, or overwrite specified. Running preprocessing.")
        processor.process(cutoff=args.cutoff)
        processor.save(outpath)
    else:
        logging.info("Loading detected cache file: "+outpath)
        processor.load(outpath)    

    logging.info("Batching training examples...")
    processor.make_dataset(
        SRC_MAX=args.max_seq_length, 
        NXT_MAX=args.max_seq_length, 
        pad_idx=tokenizer.pad_token_id, 
        cutoff=args.cutoff, 
        group_by_nxt=args.group_by_nxt)

    loader = Loader(processor, batch_size=args.batch_size, shuffle=args.shuffle)
    
    return loader


if __name__ == "__main__":
    main()