"""Check what's stored in the database."""
import sqlite3
from pathlib import Path

db_path = Path("face_authorized.sqlite3")

if not db_path.exists():
    print("❌ Database doesn't exist!")
    exit()

con = sqlite3.connect(str(db_path))

# Check persons
print("=" * 60)
print("PERSONS IN DATABASE:")
print("=" * 60)
cursor = con.execute("SELECT id, name FROM persons")
persons = cursor.fetchall()
for pid, name in persons:
    print(f"  ID: {pid}, Name: {name}")

# Check templates
print("\n" + "=" * 60)
print("TEMPLATES (Embeddings):")
print("=" * 60)
cursor = con.execute("""
    SELECT p.name, COUNT(t.id) as template_count
    FROM persons p
    LEFT JOIN templates t ON p.id = t.person_id
    GROUP BY p.name
""")
for name, count in cursor:
    print(f"  {name}: {count} embeddings")

# Check if images table exists
cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'")
has_images_table = cursor.fetchone() is not None

if has_images_table:
    # Check schema
    cursor = con.execute("PRAGMA table_info(images)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'image_path' in columns:
        print("\n" + "=" * 60)
        print("IMAGES:")
        print("=" * 60)
        cursor = con.execute("""
            SELECT p.name, i.stage, i.image_path
            FROM persons p
            LEFT JOIN images i ON p.id = i.person_id
            WHERE i.id IS NOT NULL
            ORDER BY p.name, i.stage
        """)
        rows = cursor.fetchall()
        if rows:
            for name, stage, path in rows:
                if path:
                    exists = "✓" if Path(path).exists() else "✗ (missing)"
                    print(f"  {name} - {stage}: {path} {exists}")
                else:
                    print(f"  {name} - {stage}: (no path)")
        else:
            print("  No image records found")
    else:
        print("\n⚠ Images table exists but missing image_path column")
        print("  Run: python migrate_database.py")
else:
    print("\n⚠ No 'images' table in database")

con.close()

print("\n" + "=" * 60)
print("FILESYSTEM CHECK:")
print("=" * 60)
auth_dir = Path("authorized_faces")
if auth_dir.exists():
    folders = list(auth_dir.iterdir())
    if folders:
        for folder in folders:
            if folder.is_dir():
                image_count = len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.png")))
                print(f"  {folder.name}: {image_count} images")
    else:
        print("  ⚠ authorized_faces is EMPTY")
else:
    print("  ⚠ authorized_faces folder doesn't exist")
print("=" * 60)