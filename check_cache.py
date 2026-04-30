from tinygrad.helpers import db_connection, CACHEDB
print('CACHEDB:', CACHEDB)
conn = db_connection()
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print('Tables:', tables)
for (t,) in tables:
    count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    print(f'  {t}: {count} rows')
