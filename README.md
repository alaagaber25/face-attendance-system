# Face Attendance System - Modular Architecture

A reorganized, modular face recognition attendance system with clean separation of concerns.

## 🏗️ Architecture Overview

```
face_attendance_system/
├── config.py                 # ⚙️ All configuration constants
├── models/                   # 🤖 ML models (easy to swap)
│   ├── detector.py          #    - Face detection (Haar Cascades)
│   ├── recognizer.py        #    - Face recognition (DeepFace)
│   └── tracker.py           #    - Face tracking (OpenCV trackers)
├── database/                 # 💾 Data persistence
│   └── db_manager.py        #    - SQLite operations
├── services/                 # 🔧 Business logic
│   ├── tts_service.py       #    - Text-to-speech
│   ├── enrollment_service.py#    - Multi-stage enrollment
│   └── attendance_service.py#    - Attendance logging
├── ui/                       # 🎨 User interface
│   └── display.py           #    - Overlays and visualization
├── utils/                    # 🛠️ Utilities
│   └── state.py             #    - Global state management
└── main.py                   # 🚀 Application orchestration
```

## 🎯 Key Benefits

### 1. **Easy Model Replacement**

Want to replace DeepFace with InsightFace? Just modify `models/recognizer.py`:

```python
# models/recognizer.py
class FaceRecognizer:
    def __init__(self):
        # Old: DeepFace
        # self.model = DeepFace.build_model("Facenet")
        
        # New: InsightFace
        from insightface.app import FaceAnalysis
        self.model = FaceAnalysis()
```

Want to replace Haar Cascades with YOLO? Just modify `models/detector.py`:

```python
# models/detector.py
class FaceDetector:
    def __init__(self):
        # Old: Haar Cascades
        # self.frontal_cascade = cv2.CascadeClassifier(...)
        
        # New: YOLO
        from ultralytics import YOLO
        self.model = YOLO('yolov8n-face.pt')
```

### 2. **Clear Separation of Concerns**

- **Models**: Only ML/CV algorithms
- **Services**: Only business logic
- **Database**: Only data persistence
- **UI**: Only visualization
- **State**: Only state management

### 3. **Testability**

Each module can be tested independently:

```python
# Test detector
detector = FaceDetector()
faces = detector.detect_multiple_faces(test_image)
assert len(faces) > 0

# Test recognizer
recognizer = FaceRecognizer()
embedding = recognizer.extract_embedding(face_crop)
assert embedding is not None

# Test database
db = DatabaseManager()
db.add_person("John Doe")
assert db.get_person_id("John Doe") is not None
```

### 4. **Configuration in One Place**

All settings in `config.py`:

```python
# Change detection model
MODEL_NAME = "ArcFace"  # was "Facenet"

# Change thresholds
ACCEPT_DIST_THRESH = 0.30  # was 0.35

# Change tracker
TRACKER_TYPE = "CSRT"  # was "KCF"
```

## 📦 Module Details

### `config.py`
- All constants in one place
- Mode-dependent settings (FAST_MODE vs. normal)
- Easy to tune thresholds and parameters

### `models/detector.py`
**Purpose**: Detect faces in images

**Main Classes**:
- `FaceDetector`: Abstract face detection interface

**Key Methods**:
- `detect_largest_face()`: Find primary face
- `detect_multiple_faces()`: Find up to N faces
- `detect_for_stage()`: Stage-aware detection (front/profile/left)
- `is_crop_usable()`: Quality check for face crops

**Easy to Replace**: Swap Haar Cascades with YOLO, RetinaFace, MediaPipe, etc.

### `models/recognizer.py`
**Purpose**: Extract embeddings and match identities

**Main Classes**:
- `FaceRecognizer`: Abstract recognition interface
- `RecognitionWorker`: Threaded worker for non-blocking recognition

**Key Methods**:
- `extract_embedding()`: Get face embedding
- `match_identity()`: Match against database
- `cosine_distance()`: Calculate similarity
- `is_confident_match()`: Check match confidence

**Easy to Replace**: Swap DeepFace with InsightFace, FaceNet, ArcFace, etc.

### `models/tracker.py`
**Purpose**: Track faces between frames

**Main Classes**:
- `FaceTracker`: Wrapper for OpenCV trackers

**Key Methods**:
- `init()`: Initialize with bbox
- `update()`: Update with new frame
- `reset()`: Clear tracker

**Utility Functions**:
- `ema()`: Exponential moving average for smoothing
- `ema_bbox()`: EMA for bounding boxes

### `database/db_manager.py`
**Purpose**: All SQLite operations

**Main Classes**:
- `DatabaseManager`: Centralized database interface

**Key Methods**:
- Person: `add_person()`, `get_person_id()`
- Templates: `add_template()`, `get_all_templates()`
- Images: `add_image()`
- Attendance: `log_attendance()`
- New Users: `log_new_user()`

### `services/tts_service.py`
**Purpose**: Text-to-speech with caching

**Main Classes**:
- `TTSService`: TTS with de-duplication

**Key Methods**:
- `speak()`: Non-blocking cached TTS

**Features**:
- Caches MP3 files to avoid regeneration
- De-duplicates same phrases within time window
- Non-blocking (runs in background thread)

### `services/enrollment_service.py`
**Purpose**: Multi-stage face capture for enrollment

**Main Classes**:
- `EnrollmentService`: Enrollment orchestration

**Key Methods**:
- `start_capture()`: Begin multi-stage capture
- `finalize_enrollment()`: Process and add to database
- `process_pending_enrollments()`: Handle queue

**Features**:
- Multi-stage capture (front, right, left)
- Duplicate detection
- Quality checks
- Name prompting with timeout

### `services/attendance_service.py`
**Purpose**: Attendance logging with business rules

**Main Classes**:
- `AttendanceService`: Attendance logic

**Key Methods**:
- `log_event()`: Log Arrival/Departure
- `determine_status()`: Auto-determine Arrival vs. Departure
- `process_recognition_result()`: Handle recognition → attendance

**Business Rules**:
- Only 1 Arrival + 1 Departure per person per day
- Requires stable recognition (N consecutive frames)
- 45-second cooldown between events

### `ui/display.py`
**Purpose**: All visualization and overlays

**Main Classes**:
- `DisplayManager`: UI rendering

**Key Methods**:
- `draw_primary_face_box()`: Main face box
- `draw_secondary_faces()`: Additional faces
- `draw_stats()`: Performance stats
- `draw_ui_overlays()`: Banners, countdowns, flashes

### `utils/state.py`
**Purpose**: Centralized global state

**Main Classes**:
- `ApplicationState`: Thread-safe state manager

**Key Features**:
- Database templates
- Recognition state (identity, streaks)
- Smoothed values (distance, bbox)
- Display state (colors, labels)
- Capture state
- UI state (banners, countdowns)

**Key Methods**:
- Database: `set_db_templates()`, `get_db_templates()`, `request_db_reload()`
- Capture: `is_capture_in_progress()`, `set_latest_frame()`
- UI: `show_banner()`, `start_countdown()`, `flash_capture()`
- Streaks: `update_streak()`, `reset_streak_for_unseen_ids()`

### `main.py`
**Purpose**: Orchestrate all components

**Main Classes**:
- `FaceAttendanceSystem`: Main application

**Key Methods**:
- `run()`: Main loop
- `_process_frame()`: Frame processing
- `_run_recognition()`: Heavy detection + recognition
- `_run_tracker()`: Lightweight tracking
- `_handle_key()`: Keyboard input

## 🔄 Data Flow

```
Camera Frame
    ↓
[FaceDetector]
    ↓
Detected Faces (bboxes)
    ↓
[FaceRecognizer]
    ↓
Embeddings + Distances
    ↓
[AttendanceService] ← [ApplicationState]
    ↓
[DatabaseManager]
    ↓
SQLite Database
```

## 🔧 How to Extend

### Add a New Model

1. **Create model wrapper** in `models/`
2. **Implement standard interface** (detect/recognize/track)
3. **Update imports** in `main.py`

Example - Add RetinaFace detector:

```python
# models/detector.py
from retinaface import RetinaFace

class FaceDetector:
    def __init__(self, backend="retinaface"):
        if backend == "retinaface":
            # Use RetinaFace
            self.backend = "retinaface"
        else:
            # Use Haar Cascades
            self.backend = "haar"
            self.frontal_cascade = cv2.CascadeClassifier(...)
    
    def detect_largest_face(self, bgr):
        if self.backend == "retinaface":
            faces = RetinaFace.detect_faces(bgr)
            # Process RetinaFace output...
        else:
            # Original Haar Cascade logic...
```

### Add a New Service

1. **Create service file** in `services/`
2. **Inject dependencies** (db, tts, state)
3. **Use in main.py**

Example - Add visitor logging:

```python
# services/visitor_service.py
class VisitorService:
    def __init__(self, db_manager, tts_service, state):
        self.db = db_manager
        self.tts = tts_service
        self.state = state
    
    def log_visitor(self, photo, timestamp):
        # Log unknown visitors
        pass

# main.py
self.visitor = VisitorService(self.db_manager, self.tts, self.state)
```

## 🚀 Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Run system
python main.py
```

## ⌨️ Keyboard Controls

- **'g'**: Start enrollment (capture new user)
- **'q'**: Quit application

## 📊 Features

- ✅ Multi-face detection and recognition
- ✅ Real-time tracking between recognition runs
- ✅ Multi-stage enrollment (front, right, left views)
- ✅ SQLite database for templates and attendance
- ✅ Arabic TTS feedback
- ✅ One Arrival + One Departure per person per day
- ✅ Duplicate detection during enrollment
- ✅ Quality checks for face crops
- ✅ Smoothed bounding boxes and distances
- ✅ Configurable thresholds and parameters

## 🔒 Original Logic Preserved

All original logic, models, and implementations are preserved:
- Same DeepFace model (Facenet/Facenet512)
- Same Haar Cascade detection
- Same recognition thresholds
- Same multi-stage enrollment
- Same attendance rules
- Same TTS behavior

**Only the organization changed, not the behavior!**
