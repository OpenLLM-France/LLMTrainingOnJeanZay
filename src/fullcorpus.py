import pyarrow.parquet as pq
import sys
import pickle
import time

"""
TODO:
- save into 200 files (1000 files max) pour que le "seek" soit plus rapide (??)
  note: sous linux ext4, seek() est en O(1) donc le python seek() est aussi en O(1)
  cf https://stackoverflow.com/questions/51801213/complexity-of-f-seek-in-python
- pas besoin de arrow, python seek() est O(1)
"""

t0 = time.time()
with open("alldata/all.txt","w") as f:
    with open("fnoms","r") as g:
        for l in g:
            fnom = l.strip()
            pf = pq.ParquetFile(fnom)
            t1 = time.time()
            print("FFNOM__",fnom,f.tell(), t1-t0)
            t0 = t1
            for data in pf.iter_batches():
                txtcol = [str(x) for x in data.column_names if 'text' in x or 'txt' in x]
                for x in data[txtcol[0]]:
                    f.write(str(x)+'\n')

