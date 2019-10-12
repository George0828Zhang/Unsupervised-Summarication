#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ELMo import getELMo
import json
import math
from tqdm.auto import tqdm, trange
import torch.nn.functional as F
import torch
import numpy as np


# In[2]:


data_dir = "/hdd/unsupervised-summarization/data-giga/"
vocab_path = data_dir + "vocab.json"
device = torch.device("cuda")
vocab = json.load(open(vocab_path))


# In[ ]:


elmo = getELMo(vocab, unidir=False, downstream=False, mix_parameters=[1,-9e10,-9e10])
def embed(t):
    dummy = torch.zeros((t.shape[0], t.shape[1], 50)).type_as(t)        
    embeddings = elmo(dummy, word_inputs=t)
    return embeddings['elmo_representations'][0]
elmo.to(device)
elmo.eval()
# In[ ]:


print("building candidate mapping")
# (vocab, emb)
vocab_size = len(vocab)
batch_size = 128
embeddings = []
total = int(math.ceil(vocab_size/batch_size))
progress = tqdm(range(total), total=total)
for i in progress:
    fr = i*batch_size
    to = min(fr+batch_size, vocab_size)
    indices = torch.arange(fr, to).unsqueeze(1)
    embeddings.append(embed(indices.to(device)) )

# (n_batch, batch_size)
embeddings = torch.cat(embeddings, dim=0)
print(embeddings.shape)
# (vocab, emb)


# In[ ]:


np.save(data_dir+"embeddings.npy", embeddings.squeeze().cpu().numpy())

