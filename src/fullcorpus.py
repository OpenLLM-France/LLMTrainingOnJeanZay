import pyarrow.parquet as pq
import sys
import pickle

"""
TODO:
- save into 200 files (1000 files max) pour que le "seek" soit plus rapide (??)
  note: sous linux ext4, seek() est en O(1) donc le python seek() est aussi en O(1)
  cf https://stackoverflow.com/questions/51801213/complexity-of-f-seek-in-python
- pas besoin de arrow, python seek() est O(1)
"""

with open("fnoms","r") as f: fnoms = [x.split("\t")[0] for x in f]
for fnom in fnoms:
    pf = pq.ParquetFile(fnom)
    ff = fnom+".txt"
    with open(ff+".idx","wb") as fi:
        with open(ff,"w") as f:
            for data in pf.iter_batches():
                txtcol = [str(x) for x in data.column_names if 'text' in x or 'txt' in x]
                print(txtcol)
                for x in data[txtcol[0]]:
                    f.write(str(x)+'\n')
                    pickle.dump(f.tell(),fi)
                print("FFNOM__",fnom,len(data))

