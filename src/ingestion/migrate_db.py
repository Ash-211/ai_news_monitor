import sqlite3
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
db_file = os.path.join(base_dir, 'data', 'database.sqlite')

conn = sqlite3.connect(db_file)
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE articles ADD COLUMN score_details TEXT;")
    print("Successfully added score_details column.")
except sqlite3.OperationalError as e:
    print(f"Column might already exist or error: {e}")

conn.commit()
conn.close()
