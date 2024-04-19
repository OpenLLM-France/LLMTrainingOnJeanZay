import pickle

# 23738037750 lignes
# 542859000 en 2h = 2% ==> 100h = 4 jours

with open("alldata/all.idx","wb") as gidx:
    with open("alldata/all.txt","r") as f:
        i=0
        l=f.readline()
        while l:
            pickle.dump(f.tell(),gidx)
            if i%1000==0: print("l",i)
            l = f.readline()
            i+=1
