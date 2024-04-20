import time
import random
import os

# ceci permet de lire le corpus au hasard, mais les resultats sont plutot decevants: pas sur que le corpus soit tres bon!

txtf = "alldata/all.txt"
l=os.path.getsize(txtf)
sp = ord(' ')
print("filesize",l,sp)
l -= 50000

t0=time.time()
with open(txtf,"rb") as g:
    for i in range(100):
        d = random.randrange(l)
        t1=time.time()
        g.seek(d)
        t2=time.time()
        while True:
            c=g.read(1)
            if ord(c)==sp: break
        # g.read(1)
        s = g.readline()
        s = s.decode("utf-8")
        print("R",i,t2-t1,len(s),s)
