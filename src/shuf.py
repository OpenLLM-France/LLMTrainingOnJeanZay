import sqlite3

temp_db = "utts.sqlite"
connection = sqlite3.connect(temp_db)
cursor = connection.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS lines (
        line TEXT
    );
""")
inutt = False
n=0
with open("wikifr.toks","r") as fp:
    for line in fp:
        if line[0]=='[' and line[1]==' ':
            inutt = True
            utt = [int(x) for x in line[1:].split(' ') if len(x)>0]
        elif line[-2]==']' and inutt:
            utt = utt + [int(x) for x in line[:-2].split(' ') if len(x)>0]
            n+=len(utt)
            sutt = ' '.join([str(x) for x in utt])
            print("UTT",n,sutt)
            cursor.execute("INSERT INTO lines (line) VALUES (\""+sutt+"\");")
            connection.commit()
        elif inutt:
            utt = utt + [int(x) for x in line.split(' ') if len(x)>0]

with open("shuffled.toks", "w") as fp:
  for line in cursor.execute("""
      SELECT line FROM lines ORDER BY random();
      """):
      fp.write(line[0])

