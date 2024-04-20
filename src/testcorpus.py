import pickle
import time
import random

# 23738037750 lignes, il faut 90s pour en lire 38000, donc pour tout lire == 650 jours !!! 
# pickle est bcp trop lent, j'abandonne l'idee de conserver des index, mieux vaut faire un seek random direct dans le texte

idx = {0:0}
j=0
t0=time.time()
with open("alldata/all.txt","r") as g:
    with open("alldata/all.idx","rb") as f:
        while True:
            try:
                i = pickle.load(f)
                idx[j]=i
                j+=1
                if j%1000==0:
                    t1=time.time()
                    print("T",len(idx),t1-t0)
            except: break
    t1=time.time()
    print("N",len(idx),t1-t0)

    r = [i for i in range(len(idx))]
    random.shuffle(r)
    for i in r[:100]:
        t1=time.time()
        g.seek(idx[i])
        t2=time.time()
        s=g.readline()
        print("R",i,t2-t1,s)
