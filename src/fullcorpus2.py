import pickle
import time

# 23738037750 lignes
# 542859000 en 2h = 2% ==> 100h = 4 jours
# 5937339000 en 20h = 25% ==> 100h = 80h = 3.4 jours

# PB: combien de temps prend la lecture de tous les index ? et quelle RAM ?
t0 = time.time()
with open("alldata/all.idx","rb") as gidx:
    i=0
    while True:
        try:
            idx=pickle.load(gidx)
            i+=1
        except: break
t1 = time.time()
print("last record",i,idx,t1-t0)

with open("alldata/all.idx","ab") as gidx:
    with open("alldata/all.txt","r") as f:
        f.seek(idx)
        l=f.readline()
        while l:
            pickle.dump(f.tell(),gidx)
            if i%1000==0: print("l",i)
            l = f.readline()
            i+=1
