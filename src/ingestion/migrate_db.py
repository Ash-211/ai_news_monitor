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

# ── Stage 2 & 3: Image Provenance columns ────────────────────────────────────
try:
    cursor.execute("ALTER TABLE articles ADD COLUMN image_url TEXT;")
    print("Successfully added image_url column.")
except sqlite3.OperationalError as e:
    print(f"image_url: {e}")

try:
    cursor.execute("ALTER TABLE articles ADD COLUMN image_status TEXT DEFAULT 'pending';")
    print("Successfully added image_status column.")
except sqlite3.OperationalError as e:
    print(f"image_status: {e}")

try:
    cursor.execute("ALTER TABLE articles ADD COLUMN deepfake_score REAL;")
    print("Successfully added deepfake_score column.")
except sqlite3.OperationalError as e:
    print(f"deepfake_score: {e}")

conn.commit()
conn.close()

