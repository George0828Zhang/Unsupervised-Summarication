import torch
import json
import math
import sys
from tqdm.auto import tqdm, trange

import torch.nn.functional as F

from allennlp.modules.elmo import Elmo
def getELMo(vocab, unidir, downstream=False, mix_parameters=[1,1,1]):
    options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    
    vocab_to_cache=sorted(vocab.keys(), key=lambda t: vocab[t])
    if downstream:
        elmo = Elmo(options_file, weight_file, num_output_representations=1, vocab_to_cache=vocab_to_cache)
    else:
        elmo = Elmo(options_file, weight_file, num_output_representations=1, scalar_mix_parameters=mix_parameters, vocab_to_cache=vocab_to_cache)
        

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

def fix_LM(lm_path, lm_path_out, vocab, old_vocab):    
    lm = torch.load(lm_path)    

    mapping = torch.zeros(len(vocab)).long()
    for a, b in vocab.items():
        mapping[b] = old_vocab[a]

    weight2 = lm.embed.weight[mapping]
    lm.embed.weight.data = weight2

    torch.save(lm, lm_path_out)
    return lm


def MakeEmbeddings(vocab, path_out):
    device = torch.device("cuda")
    elmo = getELMo(vocab, unidir=False, downstream=False, mix_parameters=[1,-9e10,-9e10])

    def embed(t):
        dummy = torch.zeros((t.shape[0], t.shape[1], 50)).type_as(t)        
        embeddings = elmo(dummy, word_inputs=t)
        return embeddings['elmo_representations'][0]
    elmo.to(device)
    elmo.eval()


    # (vocab, emb)
    vocab_size = len(vocab)
    batch_size = 128
    embeddings = []
    total = int(math.ceil(vocab_size/batch_size))
    progress = tqdm(range(total), total=total, desc="Extracting embeddings.")
    for i in progress:
        fr = i*batch_size
        to = min(fr+batch_size, vocab_size)
        indices = torch.arange(fr, to).unsqueeze(1)
        embeddings.append(embed(indices.to(device)) )

    # (n_batch, batch_size)
    embeddings = torch.cat(embeddings, dim=0)
    print(embeddings.shape)
    # (vocab, emb)
    embeddings = embeddings.squeeze().cpu()
    torch.save(embeddings, path_out)
    return embeddings

def cleanELMo(vocab, path_out):
    elmo = getELMo(vocab, unidir=False, downstream=False, mix_parameters=[1,1,1])
    torch.save(elmo, path_out)

def get_candidate_mapping(embeddings, path_out, batch_size = 128):
    vocab_size, emb_dim = embeddings.shape
            
    total = int(math.ceil(vocab_size/batch_size))
    
    output = []
    for i in trange(total, desc="candidate-mappings"):
        # (50000, 1024)
        fr = i*batch_size
        to = min(fr+batch_size, vocab_size)
        cur = embeddings[fr:to] #(batch, 1024)
        scores = torch.matmul(cur, embeddings.transpose(0, 1)) #(batch, 50000)
        candidates = torch.argsort(scores, dim=-1, descending=True) #(batch, 50000)
        output.append(candidates[:,:50].cpu()) #(batch, 50) # send to cpu to free memory
    mappings = torch.cat(output, dim=0)
    torch.save(mappings, path_out)
    return mappings

def main(do_LM=True, do_ELMo=True, do_Embed=True, do_Candi=True):
    data_dir = "data-fixed/"
    lm_path = data_dir+"LM-check"    
    vocab_path = data_dir+"vocab.json"
    old_vocab_path = data_dir+"old_vocab.json"

    # output paths
    lm_path_out = data_dir+"LM-check-fix"
    elmo_out = data_dir+"cleanELMo"
    emb_out = data_dir+"embeddings"
    mapping_out = data_dir+"candidate_map"
    ################

    vocab = json.load(open(vocab_path, "r"))
    old_vocab = json.load(open(old_vocab_path, "r"))

    # first fix LM
    if do_LM:
        print("fixing LM...")
        fix_LM(lm_path, lm_path_out, vocab, old_vocab)

    # then make ELMo
    if do_ELMo:
        print("Making ELMo...")
        cleanELMo(vocab, elmo_out)

    # then make Embedding
    if do_Embed:
        print("Making Embedding...")
        emb = MakeEmbeddings(vocab, emb_out)

    # then make candidate
    if do_Candi:
        print("Making candidate map...")
        get_candidate_mapping(emb, mapping_out)

if __name__ == "__main__":
    main(do_LM=False, do_ELMo=False)