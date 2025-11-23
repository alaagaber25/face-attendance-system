# Database Manager Module
# All SQLite operations centralized here

import sqlite3
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from config import AUTH_DB_PATH, MAX_TEMPLATES_PER_ID, STORE_IMAGES_IN_DB


class DatabaseManager:
    """Manages all SQLite database operations."""
    
    def __init__(self, db_path: Path = AUTH_DB_PATH):
        self.db_path = db_path
        self._ensure_initialized()
    
    def _connect(self):
        """Create a database connection with optimizations."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.db_path))
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con
    
    def _ensure_initialized(self):
        """Ensure database tables exist."""
        with self._connect() as con:
            # Persons table
            con.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            
            # Templates (embeddings) table
            con.execute("""
                CREATE TABLE IF NOT EXISTS templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    emb BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE
                )
            """)
            
            # Images table (optional)
            con.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    stage TEXT,
                    img BLOB,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE
                )
            """)
            
            # Attendance table
            con.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    ts_iso TEXT NOT NULL,
                    ts_unix REAL NOT NULL,
                    distance REAL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE SET NULL
                )
            """)
            
            # New users log table
            con.execute("""
                CREATE TABLE IF NOT EXISTS new_users_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    name TEXT NOT NULL,
                    ts_iso TEXT NOT NULL,
                    ts_unix REAL NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE SET NULL
                )
            """)
            
            # Indices
            con.execute("CREATE INDEX IF NOT EXISTS idx_attendance_ts ON attendance(ts_unix)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_attendance_name ON attendance(name)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_templates_pid ON templates(person_id)")
    
    # ========== PERSON OPERATIONS ==========
    
    def get_person_id(self, name: str) -> Optional[int]:
        """Get person ID by name."""
        with self._connect() as con:
            row = con.execute("SELECT id FROM persons WHERE name=?", (name,)).fetchone()
            return int(row[0]) if row else None
    
    def add_person(self, name: str) -> int:
        """Add a new person (or get existing ID)."""
        with self._connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO persons(name, created_at) VALUES(?, ?)",
                (name, time.time())
            )
            row = con.execute("SELECT id FROM persons WHERE name=?", (name,)).fetchone()
            return int(row[0])
    
    # ========== TEMPLATE OPERATIONS ==========
    
    def add_template(self, person_id: int, embedding: np.ndarray):
        """Add a face template (embedding) for a person."""
        emb_bytes = np.asarray(embedding, dtype=np.float32).tobytes()
        with self._connect() as con:
            con.execute(
                "INSERT INTO templates(person_id, emb, created_at) VALUES(?, ?, ?)",
                (person_id, sqlite3.Binary(emb_bytes), time.time())
            )
    
    def get_all_templates(self) -> Dict[str, List[np.ndarray]]:
        """
        Load all templates from database.
        
        Returns:
            Dict mapping person names to list of embedding vectors
        """
        with self._connect() as con:
            rows = con.execute("""
                SELECT p.name, t.emb FROM templates t
                JOIN persons p ON p.id = t.person_id
            """).fetchall()
        
        templates_dict: Dict[str, List[np.ndarray]] = {}
        
        for name, emb_blob in rows:
            embedding = np.frombuffer(emb_blob, dtype=np.float32)
            # L2 normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
            templates_dict.setdefault(name, []).append(embedding)
        
        # Limit templates per person
        for name, embeddings in templates_dict.items():
            if len(embeddings) > MAX_TEMPLATES_PER_ID:
                step = max(1, len(embeddings) // MAX_TEMPLATES_PER_ID)
                templates_dict[name] = embeddings[::step][:MAX_TEMPLATES_PER_ID]
        
        return templates_dict
    
    # ========== IMAGE OPERATIONS ==========
    
    def add_image(
        self,
        person_id: int,
        stage: str,
        image_data: Optional[np.ndarray] = None,
        image_path: Optional[str] = None
    ) -> bool:
        """
        Add an image record for a person.
        
        Args:
            person_id: Person ID
            stage: Capture stage (front, right_side, left_side)
            image_data: Image array (if storing in DB)
            image_path: Image file path (if storing on filesystem)
        """
        try:
            with self._connect() as con:
                if STORE_IMAGES_IN_DB and image_data is not None:
                    # Store binary data in database
                    import cv2
                    _, buffer = cv2.imencode('.jpg', image_data)
                    blob = buffer.tobytes()
                    
                    con.execute("""
                        INSERT INTO images (person_id, stage, image_data, image_path)
                        VALUES (?, ?, ?, NULL)
                    """, (person_id, stage, blob))
                else:
                    # Store only path reference
                    con.execute("""
                        INSERT INTO images (person_id, stage, image_data, image_path)
                        VALUES (?, ?, NULL, ?)
                    """, (person_id, stage, image_path))
                
                return True
        except Exception as e:
            print(f"[DB] Error adding image: {e}")
            return False

    # ========== ATTENDANCE OPERATIONS ==========
    
    def log_attendance(
        self, 
        name: str, 
        status: str, 
        distance: Optional[float] = None
    ) -> bool:
        """
        Log attendance event (Arrival or Departure).
        Only allows one Arrival and one Departure per person per day.
        
        Returns:
            True if logged successfully, False if already exists for today
        """
        ts = datetime.now()
        ts_iso = ts.isoformat(timespec="seconds")
        ts_unix = time.time()
        today_str = ts.strftime("%Y-%m-%d")
        
        person_id = self.get_person_id(name)
        
        try:
            with self._connect() as con:
                # Check if already logged for today
                existing = con.execute("""
                    SELECT 1 FROM attendance
                    WHERE name = ? AND status = ? AND substr(ts_iso, 1, 10) = ?
                    LIMIT 1
                """, (name, status, today_str)).fetchone()
                
                if existing:
                    return False
                
                # Insert new record
                con.execute("""
                    INSERT INTO attendance(
                        person_id, name, status, ts_iso, ts_unix, distance, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    person_id,
                    name,
                    status,
                    ts_iso,
                    ts_unix,
                    None if distance is None else float(distance),
                    time.time()
                ))
                
            return True
            
        except Exception as e:
            print(f"[DB] Attendance log failed: {e}")
            return False
    
    # ========== NEW USER LOGGING ==========
    
    def log_new_user(self, name: str, person_id: Optional[int] = None):
        """Log a new user enrollment."""
        ts = datetime.now()
        ts_iso = ts.isoformat(timespec="seconds")
        ts_unix = time.time()
        
        try:
            with self._connect() as con:
                con.execute("""
                    INSERT INTO new_users_log(person_id, name, ts_iso, ts_unix)
                    VALUES (?, ?, ?, ?)
                """, (person_id, name, ts_iso, ts_unix))
            
            print(f"[DB] New user logged: {name} (ID={person_id})")
            
        except Exception as e:
            print(f"[DB] New user log failed: {e}")
