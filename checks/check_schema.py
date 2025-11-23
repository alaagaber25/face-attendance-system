import sqlite3

con = sqlite3.connect("face_authorized.sqlite3")

# Check all tables
cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables:", [t[0] for t in tables])

# Check images table schema
cursor = con.execute("PRAGMA table_info(images)")
columns = cursor.fetchall()
print("\nImages table columns:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

con.close()