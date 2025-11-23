import sqlite3

con = sqlite3.connect("face_authorized.sqlite3")

try:
    # Check if image_path column exists
    cursor = con.execute("PRAGMA table_info(images)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'image_path' not in columns:
        print("Adding image_path column...")
        con.execute("ALTER TABLE images ADD COLUMN image_path TEXT")
        con.commit()
        print("✓ Added image_path column")
    else:
        print("✓ image_path column already exists")
    
    # Verify
    cursor = con.execute("PRAGMA table_info(images)")
    columns = cursor.fetchall()
    print("\nCurrent images table schema:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
        
except Exception as e:
    print(f"Error: {e}")
finally:
    con.close()

print("\n✓ Migration complete!")