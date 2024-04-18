import sys
import os
import time
import pickle
from transformers import AutoTokenizer

"""
- 2nd step: now that all texts are in a single indexed file, we can tokenize it all
- sur prepost: process 430720000 lignes en 12h == 2% du total ! Donc il faut 50x12 = 600h == 25 jours
"""

tokenizer = AutoTokenizer.from_pretrained("/gpfsdswork/dataset/HuggingFace_Models/bigscience/bloom-7b1")
print("ISFAST",tokenizer.is_fast)
if tokenizer.pad_token is None: tokenizer.pad_token_id = 0
tokenizer.padding_side = 'left'

# TODO: speedup by tokenizing utts in parallel

with open("alldata/all.pkl","wb") as g:
    with open("alldata/all.idx","wb") as gidx:
        with open("alldata/all.txt","r") as f:
            cur = []
            for i,l in enumerate(f):
                print("line",i)
                toks = tokenizer.encode(l)
                # already concat to 2048 + index
                cur.append(toks)
                if len(cur)>=2048:
                    toks = cur[2048:]
                    cur = cur[:2048]
                    pickle.dump(g.tell(),gidx)
                    pickle.dump(cur,g)
                    cur = toks
