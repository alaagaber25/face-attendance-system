# Face Recognition Attendance System - Configuration
# All configuration constants in one place

from pathlib import Path
import cv2
# ============= GENERAL SETTINGS =============
FAST_MODE = True
CAM_INDEX = 0
MIRROR_WEBCAM = True

# ============= DIRECTORY STRUCTURE =============
AUTHORIZED_DIR = Path("authorized_faces")
CACHE_PATH = Path("attendance") / "emb_cache.pkl"
AUTH_DB_PATH = Path("face_authorized.sqlite3")
TTS_CACHE_DIR = Path("tts_cache")

# ============= DATABASE SETTINGS =============
CACHE_ENABLED = True
STORE_IMAGES_IN_DB = False

# ============= RECOGNITION THRESHOLDS =============
ACCEPT_DIST_THRESH = 0.75      # Distance threshold for accepting a match
REVOKE_DIST_THRESH = 0.70      # Distance threshold for revoking authorization
TOP2_MARGIN_MIN = 0.10         # Minimum margin between top 2 matches
TOP2_RATIO_MAX = 0.85          # Maximum ratio between top 2 distances

# ============= MODE-DEPENDENT SETTINGS =============
if FAST_MODE:
    FRAME_DOWNSCALE = 0.7      # Increased from 0.5 for better detection
    DETECT_EVERY_N_BASE = 10
    TRACKER_TYPE = "KCF"
    MODEL_NAME = "Facenet"
    MAX_TEMPLATES_PER_ID = 2
    STABLE_FRAMES_AUTH = 7
    COLOR_HOLD_FRAMES = 8
    MIN_BBOX_AREA = 50 * 50    # Reduced from 70*70 for smaller faces
    MIN_LAPLACE_VAR = 30.0     # Reduced from 40.0 for less strict blur check
else:
    FRAME_DOWNSCALE = 0.7
    DETECT_EVERY_N_BASE = 6
    TRACKER_TYPE = "CSRT"
    MODEL_NAME = "Facenet512"
    MAX_TEMPLATES_PER_ID = 4
    STABLE_FRAMES_AUTH = 7
    COLOR_HOLD_FRAMES = 12
    MIN_BBOX_AREA = 80 * 80
    MIN_LAPLACE_VAR = 45.0

# ============= RECOGNITION TIMING =============
HEAVY_MIN_PERIOD_SEC = 0.20
NO_FACE_BACKOFF_MAX_N = 24
NO_FACE_BACKOFF_STEP = 2

# ============= ENROLLMENT SETTINGS =============
STAGE_LIST = [
    ("front", 2, "استعد لوضعية الوجه الأمامي"),
    ("right_side", 2, "استعد لوضعية الجانب الأيمن"),
    ("left_side", 2, "استعد لوضعية الجانب الأيسر"),
]
PRE_CAPTURE_COOLDOWN_SEC = 10.0
STAGE_COOLDOWN_SEC = 10.0
STAGE_TIMEOUT_PER_STAGE = 25.0
CAPTURE_IMAGE_INTERVAL = 0.25
NEW_USER_PREFIX = "user_"
CAPTURE_TRIGGER_COOLDOWN_SEC = 8.0
DIRECT_ENROLL_TO_AUTH = True
NAME_PROMPT_TIMEOUT_SEC = 12.0

# ============= ENROLLMENT GATING =============
CAPTURE_ONLY_WHEN_UNAUTHORIZED = True
CAPTURE_MIN_DIST_FOR_NEW = 0.55
CAPTURE_SUPPRESS_AFTER_AUTH_SEC = 15.0

# ============= TTS SETTINGS =============
ENABLE_TTS = True
TTS_LANG = "ar"
TTS_DEDUPE_SEC = 4.0

# ============= UI SETTINGS =============
DRAW_THICKNESS = 2
DIST_SMOOTH_ALPHA = 0.30
BBOX_SMOOTH_ALPHA = 0.40
OVERLAY_TEXT = (255, 255, 255)
FONT=cv2.FONT_HERSHEY_SIMPLEX