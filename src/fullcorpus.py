import pyarrow.parquet as pq
import sys

fnom = sys.argv[1]
pf = pq.ParquetFile(fnom)
ff = fnom+".txt"
with open(ff,"w") as f:
    for data in pf.iter_batches():
        txtcol = [str(x) for x in data.column_names if 'text' in x or 'txt' in x]
        print(txtcol)
        for x in data[txtcol[0]]: f.write(str(x)+'\n')
        print("FFNOM__",fnom,len(data))

