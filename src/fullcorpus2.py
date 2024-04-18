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
            lines = []
            cur = []
            for i,l in enumerate(f):
                lines.append(l)
                if len(lines)>10:
                    ltoks = tokenizer(lines, return_tensors=None)
                    # already concat to 2048 + index
                    for toks in ltoks['input_ids']: cur.append(toks)
                    while len(cur)>=2048:
                        pickle.dump(g.tell(),gidx)
                        pickle.dump(cur[:2048],g)
                        cur = cur[2048:]
                    print("l",i,len(ltoks['input_ids']),' '.join([str(len(x)) for x in ltoks['input_ids']]))
                    gidx.flush()
                    g.flush()
                    lines = []
