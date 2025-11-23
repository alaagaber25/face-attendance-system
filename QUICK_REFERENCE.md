# Quick Reference Card

## 🚀 Common Tasks

### Start the System
```bash
cd face_attendance_system
python main.py
```

### Change Detection Model
```python
# File: models/detector.py

# Option 1: Keep Haar Cascades (current)
class FaceDetector:
    def __init__(self):
        self.frontal_cascade = cv2.CascadeClassifier(...)
        self.profile_cascade = cv2.CascadeClassifier(...)

# Option 2: Use YOLO
class FaceDetector:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO('yolov8n-face.pt')

# Option 3: Use RetinaFace
class FaceDetector:
    def __init__(self):
        from retinaface import RetinaFace
        self.model = RetinaFace

# Option 4: Use MediaPipe
class FaceDetector:
    def __init__(self):
        import mediapipe as mp
        self.face_detection = mp.solutions.face_detection.FaceDetection()
```

### Change Recognition Model
```python
# File: models/recognizer.py

# Option 1: Keep DeepFace (current)
class FaceRecognizer:
    def __init__(self):
        from deepface import DeepFace
        self.model = DeepFace.build_model("Facenet")

# Option 2: Use InsightFace
class FaceRecognizer:
    def __init__(self):
        from insightface.app import FaceAnalysis
        self.model = FaceAnalysis(name='buffalo_l')

# Option 3: Use FaceNet
class FaceRecognizer:
    def __init__(self):
        from keras_facenet import FaceNet
        self.model = FaceNet()

# Option 4: Use ArcFace
class FaceRecognizer:
    def __init__(self):
        import onnxruntime
        self.session = onnxruntime.InferenceSession("arcface.onnx")
```

### Adjust Thresholds
```python
# File: config.py

# Recognition thresholds
ACCEPT_DIST_THRESH = 0.35      # Lower = stricter (reduce false positives)
REVOKE_DIST_THRESH = 0.45      # Higher = more forgiving (keep authorization)
TOP2_MARGIN_MIN = 0.10         # Minimum gap between top 2 matches
TOP2_RATIO_MAX = 0.85          # Maximum ratio between top 2

# Quality checks
MIN_BBOX_AREA = 70 * 70        # Minimum face size (pixels²)
MIN_LAPLACE_VAR = 40.0         # Minimum sharpness (blur detection)

# Stability requirements
STABLE_FRAMES_AUTH = 7         # Consecutive frames for stable recognition
COLOR_HOLD_FRAMES = 8          # Frames to hold color/label transition
```

### Change Camera
```python
# File: config.py

CAM_INDEX = 0  # Change to 1, 2, etc. for different cameras
```

### Enable/Disable TTS
```python
# File: config.py

ENABLE_TTS = True   # Set to False to disable all TTS
TTS_LANG = "ar"     # Language code
```

### Modify Attendance Rules
```python
# File: services/attendance_service.py

def determine_status(self) -> str:
    """Determine Arrival vs Departure."""
    hour = datetime.now().hour
    
    # Current: Before 6 PM = Arrival
    return "Arrival" if hour < 18 else "Departure"
    
    # Option 2: Fixed work hours
    # if 6 <= hour < 12:
    #     return "Arrival"
    # elif 12 <= hour < 18:
    #     return "Departure"
    # else:
    #     return "After Hours"
```

### Add New Database Table
```python
# File: database/db_manager.py

class DatabaseManager:
    def _ensure_initialized(self):
        with self._connect() as con:
            # ... existing tables ...
            
            # Add new table
            con.execute("""
                CREATE TABLE IF NOT EXISTS visitors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    photo BLOB,
                    notes TEXT
                )
            """)
```

### Customize UI Colors
```python
# File: config.py

# Drawing
DRAW_THICKNESS = 2             # Bbox line thickness
OVERLAY_TEXT = (255, 255, 255) # White text

# File: ui/display.py

# Colors for authorized/unauthorized
color = (0, 200, 0)  # Green for authorized
color = (0, 0, 255)  # Red for unauthorized

# Change to other colors:
# Yellow: (0, 255, 255)
# Orange: (0, 165, 255)
# Purple: (255, 0, 255)
# Cyan: (255, 255, 0)
```

### Change Frame Processing Speed
```python
# File: config.py

if FAST_MODE:
    FRAME_DOWNSCALE = 0.5        # Lower = faster but lower quality
    DETECT_EVERY_N_BASE = 10     # Higher = faster but less responsive
else:
    FRAME_DOWNSCALE = 0.7
    DETECT_EVERY_N_BASE = 6
```

### Modify Enrollment Stages
```python
# File: config.py

STAGE_LIST = [
    ("front", 2, "استعد لوضعية الوجه الأمامي"),
    ("right_side", 2, "استعد لوضعية الجانب الأيمن"),
    ("left_side", 2, "استعد لوضعية الجانب الأيسر"),
]

# Add more stages:
STAGE_LIST = [
    ("front", 3, "Front view prompt"),
    ("right_side", 2, "Right side prompt"),
    ("left_side", 2, "Left side prompt"),
    ("smile", 1, "Smile prompt"),  # New stage
]

# Or simplify:
STAGE_LIST = [
    ("front", 4, "Front view only"),
]
```

## 📁 File Locations Cheat Sheet

| What | Where |
|------|-------|
| All settings | `config.py` |
| Face detection | `models/detector.py` |
| Face recognition | `models/recognizer.py` |
| Face tracking | `models/tracker.py` |
| Database operations | `database/db_manager.py` |
| Text-to-speech | `services/tts_service.py` |
| Enrollment logic | `services/enrollment_service.py` |
| Attendance logic | `services/attendance_service.py` |
| UI rendering | `ui/display.py` |
| Global state | `utils/state.py` |
| Main application | `main.py` |

## 🐛 Debugging Tips

### Enable Debug Output
```python
# File: models/detector.py
def detect_multiple_faces(self, bgr, max_faces=5):
    boxes = ...
    print(f"[DEBUG] Detected {len(boxes)} faces")  # Add debug prints
    return boxes
```

### Visualize Embeddings
```python
# File: models/recognizer.py
def extract_embedding(self, bgr_crop):
    embedding = ...
    print(f"[DEBUG] Embedding shape: {embedding.shape}")
    print(f"[DEBUG] Embedding norm: {np.linalg.norm(embedding):.4f}")
    return embedding
```

### Check Database
```bash
sqlite3 face_authorized.sqlite3
> SELECT * FROM persons;
> SELECT COUNT(*) FROM templates;
> SELECT * FROM attendance WHERE date(ts_iso) = date('now');
> .quit
```

### Monitor State
```python
# File: main.py, in run() method
def run(self):
    while True:
        # ... existing code ...
        
        # Add debug output
        if frame_idx % 30 == 0:  # Every 30 frames
            print(f"[STATE] identity={self.state.current_identity}, "
                  f"streak={self.state.auth_streak}, "
                  f"dist={self.state.smoothed_distance:.3f if self.state.smoothed_distance else None}")
```

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `g` | Start enrollment (capture new user) |
| `q` | Quit application |

## 📊 Performance Tuning

### For Speed (sacrifice accuracy)
```python
# config.py
FAST_MODE = True
FRAME_DOWNSCALE = 0.3          # Lower resolution
DETECT_EVERY_N_BASE = 15       # Less frequent detection
MIN_BBOX_AREA = 50 * 50        # Smaller faces OK
```

### For Accuracy (sacrifice speed)
```python
# config.py
FAST_MODE = False
FRAME_DOWNSCALE = 0.8          # Higher resolution
DETECT_EVERY_N_BASE = 3        # More frequent detection
MIN_BBOX_AREA = 100 * 100      # Larger faces only
MIN_LAPLACE_VAR = 60.0         # Stricter quality
```

### For Multi-Face Performance
```python
# main.py, in _run_recognition()
results = self.recognition_worker.get_result()

# Limit number of faces processed
max_faces = 3  # Instead of default 5
```

## 🔧 Common Modifications

### Add Logging
```python
# Add to any file
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use instead of print
logger.info("Detection complete")
logger.warning("No face detected")
logger.error("Recognition failed")
```

### Add Configuration File
```python
# config_loader.py
import yaml

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

# config.yaml
detection:
  backend: yolo
  threshold: 0.5
recognition:
  backend: insightface
  model: buffalo_l
thresholds:
  accept: 0.35
  revoke: 0.45
```

### Add Web Interface
```python
# app.py (Flask)
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
system = FaceAttendanceSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        system.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
```

## 📚 Learn More

- **README.md** - Full documentation
- **MIGRATION_GUIDE.md** - Detailed before/after comparison
- **MODEL_SWAPPING_GUIDE.md** - Complete examples for model replacement
- **ARCHITECTURE.md** - Visual diagrams and design patterns

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Can't import modules | Make sure you're in `face_attendance_system/` directory |
| Camera won't open | Try different `CAM_INDEX` values (0, 1, 2, ...) |
| Slow performance | Enable `FAST_MODE=True`, reduce `FRAME_DOWNSCALE` |
| False positives | Lower `ACCEPT_DIST_THRESH`, increase `STABLE_FRAMES_AUTH` |
| False negatives | Raise `ACCEPT_DIST_THRESH`, lower `STABLE_FRAMES_AUTH` |
| TTS not working | Check `playsound` installation, verify audio output |

---

**Quick Start**: `cd face_attendance_system && python main.py`  
**Questions?** Check the detailed documentation files! 📚
