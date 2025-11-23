# Architecture Diagram

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (Orchestrator)                  │
│                    FaceAttendanceSystem Class                   │
└────────────┬────────────────────────────────────────────────────┘
             │
             │ coordinates
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌─────────┐     ┌─────────┐
│ Camera  │     │ Display │
│ Input   │     │ Output  │
└─────────┘     └─────────┘
```

## Detailed Component Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                            Application Layer                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  main.py - FaceAttendanceSystem                                │ │
│  │  • Initialization                                              │ │
│  │  • Main loop                                                   │ │
│  │  • Frame processing                                            │ │
│  │  • Keyboard handling                                           │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               │ uses
                               │
┌──────────────────────────────┼───────────────────────────────────────┐
│                    Service Layer                                     │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ AttendanceService│  │ EnrollmentService│  │   TTSService     │   │
│  │ • log_event()   │  │ • start_capture()│  │   • speak()      │   │
│  │ • determine_    │  │ • finalize_      │  │   • cache TTS    │   │
│  │   status()      │  │   enrollment()   │  │                  │   │
│  └─────────────────┘  └──────────────────┘  └──────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               │ uses
                               │
┌──────────────────────────────┼───────────────────────────────────────┐
│                      Model Layer (ML/CV)                             │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  FaceDetector   │  │  FaceRecognizer  │  │   FaceTracker    │   │
│  │ • detect_       │  │ • extract_       │  │   • init()       │   │
│  │   largest_face()│  │   embedding()    │  │   • update()     │   │
│  │ • detect_       │  │ • match_         │  │   • reset()      │   │
│  │   multiple_     │  │   identity()     │  │                  │   │
│  │   faces()       │  │ • cosine_        │  │                  │   │
│  │                 │  │   distance()     │  │                  │   │
│  └─────────────────┘  └──────────────────┘  └──────────────────┘   │
│        │                      │                                      │
│        │ Haar Cascades        │ DeepFace/Facenet                     │
│        │ (swappable to        │ (swappable to                        │
│        │  YOLO, RetinaFace,   │  InsightFace,                        │
│        │  MediaPipe)          │  ArcFace, etc.)                      │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               │ uses
                               │
┌──────────────────────────────┼───────────────────────────────────────┐
│                      Data Layer                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │            DatabaseManager                               │       │
│  │  • Person ops:  add_person(), get_person_id()           │       │
│  │  • Template ops: add_template(), get_all_templates()    │       │
│  │  • Image ops:   add_image()                             │       │
│  │  • Attendance:  log_attendance()                        │       │
│  │  • New users:   log_new_user()                          │       │
│  └──────────────────────────────────────────────────────────┘       │
│                               │                                      │
│                               ▼                                      │
│                    ┌──────────────────┐                             │
│                    │  SQLite Database │                             │
│                    │  • persons       │                             │
│                    │  • templates     │                             │
│                    │  • images        │                             │
│                    │  • attendance    │                             │
│                    │  • new_users_log │                             │
│                    └──────────────────┘                             │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               │ uses
                               │
┌──────────────────────────────┼───────────────────────────────────────┐
│                   State Management Layer                             │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │            ApplicationState (Singleton)                  │       │
│  │  • db_templates: Dict[str, List[np.ndarray]]            │       │
│  │  • current_identity: Optional[str]                       │       │
│  │  • auth_streak: int                                      │       │
│  │  • per_id_streak: Dict[str, int]                        │       │
│  │  • smoothed_distance: Optional[float]                    │       │
│  │  • smoothed_bbox: Optional[Tuple]                        │       │
│  │  • capture_in_progress: bool                             │       │
│  │  • ui_state: countdown, banners, flash                   │       │
│  │  • Methods: thread-safe getters/setters                  │       │
│  └──────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               │ used by
                               │
┌──────────────────────────────┼───────────────────────────────────────┐
│                       UI Layer                                       │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │            DisplayManager                                │       │
│  │  • draw_primary_face_box()                              │       │
│  │  • draw_secondary_faces()                               │       │
│  │  • draw_stats()                                         │       │
│  │  • draw_ui_overlays()                                   │       │
│  │    - Countdown timer                                    │       │
│  │    - Capture flash                                      │       │
│  │    - Name prompt banner                                 │       │
│  │    - General banners                                    │       │
│  └──────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  OpenCV      │
                        │  cv2.imshow()│
                        └──────────────┘
```

## Data Flow Diagram

```
┌──────────┐
│  Camera  │
│  Frame   │
└─────┬────┘
      │
      ▼
┌──────────────────┐
│  FaceDetector    │
│  detect_multiple │
│  _faces()        │
└─────┬────────────┘
      │
      │ List of bboxes
      │
      ▼
┌──────────────────────┐
│  For each bbox:      │
│  FaceRecognizer      │
│  extract_embedding() │
└─────┬────────────────┘
      │
      │ Embeddings
      │
      ▼
┌───────────────────────┐
│  FaceRecognizer       │
│  match_identity()     │
│  (compare with DB)    │
└─────┬─────────────────┘
      │
      │ (identity, distance, confidence)
      │
      ▼
┌─────────────────────────┐
│  ApplicationState       │
│  update_streak()        │
│  (track stability)      │
└─────┬───────────────────┘
      │
      │ If streak stable
      │
      ▼
┌──────────────────────────┐
│  AttendanceService       │
│  process_recognition_    │
│  result()                │
└─────┬────────────────────┘
      │
      │ If should log
      │
      ▼
┌──────────────────────────┐
│  DatabaseManager         │
│  log_attendance()        │
└─────┬────────────────────┘
      │
      ▼
┌──────────────┐
│  SQLite DB   │
│  attendance  │
│  table       │
└──────────────┘
      │
      ├───────────────────┐
      │                   │
      ▼                   ▼
┌──────────────┐   ┌─────────────┐
│  TTSService  │   │ DisplayMgr  │
│  speak()     │   │ show_banner │
└──────────────┘   └─────────────┘
```

## Enrollment Flow

```
User presses 'g'
      │
      ▼
┌─────────────────────┐
│  Check if allowed   │
│  - Not in progress? │
│  - Cooldown passed? │
│  - Face unknown?    │
└─────┬───────────────┘
      │ Yes
      ▼
┌──────────────────────────┐
│  EnrollmentService       │
│  start_capture()         │
│  (background thread)     │
└─────┬────────────────────┘
      │
      ▼
┌──────────────────────────┐
│  Multi-stage capture:    │
│  1. Front view (2 imgs)  │
│  2. Right view (2 imgs)  │
│  3. Left view (2 imgs)   │
└─────┬────────────────────┘
      │
      ▼
┌──────────────────────────┐
│  Prompt for name         │
│  (12 sec timeout)        │
└─────┬────────────────────┘
      │
      ▼
┌──────────────────────────┐
│  For each image:         │
│  - Detect face           │
│  - Extract embedding     │
│  - Save to DB            │
└─────┬────────────────────┘
      │
      ▼
┌──────────────────────────┐
│  DatabaseManager         │
│  - add_person()          │
│  - add_template()        │
│  - log_new_user()        │
└─────┬────────────────────┘
      │
      ▼
┌──────────────────────────┐
│  Request DB reload       │
└──────────────────────────┘
```

## Module Dependencies

```
main.py
  │
  ├─→ config.py (constants)
  │
  ├─→ models/
  │   ├─→ detector.py
  │   ├─→ recognizer.py
  │   └─→ tracker.py
  │
  ├─→ database/
  │   └─→ db_manager.py
  │
  ├─→ services/
  │   ├─→ tts_service.py
  │   ├─→ attendance_service.py ──→ db_manager, tts_service, state
  │   └─→ enrollment_service.py ──→ detector, recognizer, db_manager, tts, state
  │
  ├─→ ui/
  │   └─→ display.py ──→ state
  │
  └─→ utils/
      └─→ state.py (singleton)
```

## Key Design Patterns Used

1. **Singleton Pattern**: `ApplicationState` (single global instance)
2. **Dependency Injection**: Services receive dependencies via constructor
3. **Strategy Pattern**: Detector/Recognizer can be swapped
4. **Observer Pattern**: State changes trigger UI updates
5. **Factory Pattern**: Tracker creation with fallbacks
6. **Service Layer**: Business logic separated from infrastructure
7. **Repository Pattern**: DatabaseManager abstracts data access

## Thread Safety

```
Main Thread:
  ├─ Camera capture
  ├─ Frame processing
  ├─ UI rendering
  ├─ Keyboard handling
  └─ State updates (with locks)

Recognition Worker Thread:
  ├─ Heavy face detection
  ├─ Embedding extraction
  ├─ Identity matching
  └─ Result storage (with locks)

Enrollment Worker Thread:
  ├─ Multi-stage capture
  ├─ Image saving
  ├─ Name prompting
  └─ Database insertion (with locks)

TTS Worker Threads:
  └─ Audio playback (multiple concurrent)

Locks Used:
  • state.capture_lock
  • state.latest_frame_lock
  • state.pending_staging_lock
  • state.ui_lock
  • state.db_reload_lock
```

---

**Benefits of This Architecture:**

✅ **Clear Separation**: Each layer has distinct responsibilities  
✅ **Easy Testing**: Mock dependencies for unit tests  
✅ **Maintainable**: Changes isolated to specific modules  
✅ **Extensible**: Add new features without modifying existing code  
✅ **Swappable**: Replace models by editing one file  
✅ **Thread-Safe**: Proper locking for concurrent operations  
✅ **Documented**: Clear interfaces and responsibilities
