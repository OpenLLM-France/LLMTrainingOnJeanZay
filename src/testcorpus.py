import pickle

idx = {0:0}
j=0
with open("alldata/all.txt","r") as g:
    with open("alldata/all.idx","rb") as f:
        while True:
            try:
                i = pickle.load(f)
                idx[j]=i
                j+=1
            except: break
            g.seek(idx[j-1])
            s=g.readline()
            print("N",len(idx),s)
