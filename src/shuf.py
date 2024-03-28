import sqlite3

temp_db = tempfile.NamedTemporaryFile(delete=False)
connection = sqlite3.connect(temp_db.name)
cursor = connection.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS lines (
        line TEXT
    );
""")
with open(filename_in) as fp:
    line = fp.readline()
    while line:
        cursor.execute("INSERT INTO lines (line) VALUES (?);", [line])
        line = fp.readline()
    connection.commit()
with open(filename_out, "w") as fp:
  for line in cursor.execute("""
      SELECT line FROM lines ORDER BY random();
      """):
      fp.write(line[0])
