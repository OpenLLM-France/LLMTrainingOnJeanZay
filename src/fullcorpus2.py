import sys
import os
import time
import pickle
from transformers import AutoTokenizer
from datasets import load_dataset

"""
# peut-etre utiliser la classe Dataset pour tokeniser en multithread ? cf. ci-dessous

data_files = {'train': 'alldata/all.txt'}
dataset = load_dataset('text', data_files=data_files, split='train', streaming=True)
def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')
dataset = dataset.map(encode, batched=True, remove_columns=["text", "timestamp", "url"])
"""


"""
- 2nd step: now that all texts are in a single indexed file, we can tokenize it all
- sur prepost: process 430720000 lignes en 12h == 2% du total 23738037750 ! Donc il faut 50x12 = 600h == 25 jours
- avec des "batch" de 1000 pour le tokenizer: 982439457 en 10h30 ==> 4% du total, donc il faut 11 jours
- autre option: revenir a une tokenization pendant le train et simplement indexer ici les lignes
"""

tokenizer = AutoTokenizer.from_pretrained("/gpfsdswork/dataset/HuggingFace_Models/bigscience/bloom-7b1")
print("ISFAST",tokenizer.is_fast)
if tokenizer.pad_token is None: tokenizer.pad_token_id = 0
tokenizer.padding_side = 'left'

# TODO: speedup by tokenizing utts in parallel

t0 = time.time()
ttok = 0
tsav = 0
nseq = 0
with open("alldata/all.pkl","wb") as g:
    with open("alldata/all.idx","wb") as gidx:
        with open("alldata/all.txt","r") as f:
            lines = []
            cur = []
            for i,l in enumerate(f):
                lines.append(l)
                if len(lines)>1000:
                    t1 = time.time()
                    ltoks = tokenizer(lines, return_tensors=None)
                    t2 = time.time()
                    ttok += t2-t1
                    # already concat to 2048 + index
                    for toks in ltoks['input_ids']: cur.append(toks)
                    while len(cur)>=2048:
                        t1 = time.time()
                        nseq += 1
                        pickle.dump(g.tell(),gidx)
                        pickle.dump(cur[:2048],g)
                        t2 = time.time()
                        tsav += t2-t1
                        cur = cur[2048:]
                    mtokperh = (float(nseq)*3.600*2.048)/float(t2-t0)
                    # millions of tokens per hour
                    # la majorite du temps est pris par le tokenizer: 760s vs. 21s pour le save pour un total de 840s
                    # PB1: trop lent: 61Mtoks/h alors que HF tokenizers fast run a 50Mtoks/s
                    # PB2: la vitesse decroit lentement
                    print("l",i,mtokperh,ttok,tsav,len(ltoks['input_ids']))
                    gidx.flush()
                    g.flush()
                    lines = []
