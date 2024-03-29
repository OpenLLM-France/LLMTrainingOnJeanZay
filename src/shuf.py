import sqlite3
import time

temp_db = "/home/xtof/nvme/openllm/utts.sqlite"
connection = sqlite3.connect(temp_db)
cursor = connection.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS lines (
        line TEXT
    );
""")
n=0
putt=[]
t0=time.time()
with open("/home/xtof/nvme/openllm/wikifr.toks","r") as fp:
    for line in fp:
        if line[0]=='Y':
            utt = [int(x) for x in line[2:].split(' ') if len(x)>0]
            n+=len(utt)
            t1=time.time()
            print("NTOKS",n,t1-t0)
            if len(utt)>0: putt += utt
            while len(putt)>=2048:
                sutt = ' '.join([str(x) for x in putt[:2048]])
                cursor.execute("INSERT INTO lines (line) VALUES (\""+sutt+"\");")
                connection.commit()
                putt = putt[2048:]

with open("/home/xtof/nvme/openllm/shuffled.toks", "w") as fp:
  for line in cursor.execute("""
      SELECT line FROM lines ORDER BY random();
      """):
      fp.write(line[0])

