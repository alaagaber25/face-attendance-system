# State Management Module
# Centralized global state for the application

import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

from config import STABLE_FRAMES_AUTH, COLOR_HOLD_FRAMES


class ApplicationState:
    """Centralized application state manager."""
    
    def __init__(self):
        # Database templates
        self.db_templates: Dict[str, List[np.ndarray]] = {}
        self.db_reload_requested = False
        self.db_reload_lock = threading.Lock()
        
        # Recognition state
        self.current_identity: Optional[str] = None
        self.auth_streak = 0
        self.per_id_streak: Dict[str, int] = {}  # Per-identity streak counters
        self.last_auth_time: Dict[str, float] = {}
        self.last_auth_seen_timestamp = 0.0
        
        # Smoothed values for primary face
        self.smoothed_distance: Optional[float] = None
        self.smoothed_bbox: Optional[Tuple[float, float, float, float]] = None
        
        # Display state for primary face
        self.last_draw_color = (0, 0, 255)  # Red
        self.last_draw_label = "NOT AUTHORIZED"
        self.hold_counter = 0
        
        # Secondary faces overlay data
        self.last_secondary_overlays: List[Tuple[int, int, int, int, str, Tuple]] = []
        
        # Timing
        self.last_heavy_submit_timestamp = 0.0
        self.detect_every_n = 10  # Will be set from config
        self.no_face_cycles = 0
        self.last_detect_recog_seconds = 0.0
        self.last_auth_latency_seconds = 0.0
        
        # Pending authentication (for latency measurement)
        self.pending_auth_identity: Optional[str] = None
        self.pending_auth_start_timestamp = 0.0
        
        # Capture state
        self.capture_in_progress = False
        self.capture_lock = threading.Lock()
        self.last_capture_trigger_timestamp = 0.0
        self.last_capture_frame_index = -999999
        
        # Latest frame for capture worker
        self.latest_frame_for_capture = None
        self.latest_frame_lock = threading.Lock()
        
        # Pending enrollment staging queue
        self.pending_staging_queue = deque()
        self.pending_staging_lock = threading.Lock()
        
        # UI state
        self.ui_lock = threading.Lock()
        self.countdown_active = False
        self.countdown_end_timestamp = 0.0
        self.countdown_label = ""
        self.flash_capture_until = 0.0
        self.name_prompt_banner_until = 0.0
        self.banner_until = 0.0
        self.banner_text = ""
    
    # ========== DATABASE OPERATIONS ==========
    
    def set_db_templates(self, templates: Dict[str, List[np.ndarray]]):
        """Update the database templates."""
        self.db_templates = templates
    
    def get_db_templates(self) -> Dict[str, List[np.ndarray]]:
        """Get current database templates."""
        return self.db_templates
    
    def request_db_reload(self):
        """Request a database reload."""
        with self.db_reload_lock:
            self.db_reload_requested = True
    
    def check_and_clear_db_reload(self) -> bool:
        """Check if DB reload is requested and clear the flag."""
        with self.db_reload_lock:
            if self.db_reload_requested:
                self.db_reload_requested = False
                return True
            return False
    
    # ========== CAPTURE OPERATIONS ==========
    
    def is_capture_in_progress(self) -> bool:
        """Check if capture is currently in progress."""
        with self.capture_lock:
            return self.capture_in_progress
    
    def set_capture_in_progress(self, value: bool):
        """Set capture in progress flag."""
        with self.capture_lock:
            self.capture_in_progress = value
    
    def set_latest_frame(self, frame):
        """Update the latest frame for capture worker."""
        with self.latest_frame_lock:
            self.latest_frame_for_capture = frame.copy() if frame is not None else None
    
    def get_latest_frame(self):
        """Get a copy of the latest frame."""
        with self.latest_frame_lock:
            return self.latest_frame_for_capture.copy() if self.latest_frame_for_capture is not None else None
    
    # ========== STAGING QUEUE OPERATIONS ==========
    
    def enqueue_staging(self, folder_path):
        """Add a folder to the pending staging queue."""
        with self.pending_staging_lock:
            self.pending_staging_queue.append(folder_path)
    
    def dequeue_staging(self):
        """Remove and return a folder from the staging queue."""
        with self.pending_staging_lock:
            if self.pending_staging_queue:
                return self.pending_staging_queue.popleft()
            return None
    
    def has_pending_staging(self) -> bool:
        """Check if there are pending staging items."""
        with self.pending_staging_lock:
            return len(self.pending_staging_queue) > 0
    
    # ========== UI OPERATIONS ==========
    
    def show_banner(self, text: str, seconds: float = 2.5):
        """Show a banner message."""
        with self.ui_lock:
            self.banner_text = text
            self.banner_until = time.time() + max(1.0, seconds)
    
    def start_countdown(self, label: str, seconds: float):
        """Start a countdown timer."""
        with self.ui_lock:
            self.countdown_label = label
            self.countdown_end_timestamp = time.time() + max(1.0, seconds)
            self.countdown_active = True
    
    def stop_countdown(self):
        """Stop the countdown timer."""
        with self.ui_lock:
            self.countdown_active = False
    
    def flash_capture(self, duration: float = 0.6):
        """Flash the capture indicator."""
        with self.ui_lock:
            self.flash_capture_until = time.time() + max(0.1, duration)
    
    def show_name_prompt_banner(self, duration: float = 12.0):
        """Show the name prompt banner."""
        with self.ui_lock:
            self.name_prompt_banner_until = time.time() + max(2.0, duration)
    
    def get_ui_state(self) -> dict:
        """Get current UI state (thread-safe)."""
        with self.ui_lock:
            return {
                "countdown_active": self.countdown_active,
                "countdown_end": self.countdown_end_timestamp,
                "countdown_label": self.countdown_label,
                "flash_capture_until": self.flash_capture_until,
                "name_prompt_until": self.name_prompt_banner_until,
                "banner_until": self.banner_until,
                "banner_text": self.banner_text
            }
    
    # ========== STREAK MANAGEMENT ==========
    
    def update_streak(self, identity: str, is_authorized: bool, distance: Optional[float]) -> bool:
        """
        Update per-identity streak and determine if logging should occur.
        
        Returns:
            True if an attendance event should be logged
        """
        if not identity or identity == "Unknown":
            return False
        
        # Reset streak if not authorized
        if not is_authorized:
            self.per_id_streak[identity] = 0
            return False
        
        # Increment streak for authorized frame
        prev_streak = self.per_id_streak.get(identity, 0)
        self.per_id_streak[identity] = prev_streak + 1
        
        # Check if stable
        if self.per_id_streak[identity] < STABLE_FRAMES_AUTH:
            return False
        
        # Check cooldown
        now = time.time()
        last_auth = self.last_auth_time.get(identity, 0.0)
        if (now - last_auth) < 45.0:
            # Still in cooldown, reset streak
            self.per_id_streak[identity] = 0
            return False
        
        # Ready to log!
        self.last_auth_time[identity] = now
        self.last_auth_seen_timestamp = now
        self.per_id_streak[identity] = 0  # Reset after logging
        
        return True
    
    def reset_streak_for_unseen_ids(self, seen_ids: set):
        """Reset streaks for identities that weren't seen in current frame."""
        for identity in list(self.per_id_streak.keys()):
            if identity not in seen_ids and identity != self.current_identity:
                self.per_id_streak[identity] = 0


# Global instance
_state = ApplicationState()


# Convenience functions for accessing global state
def get_state() -> ApplicationState:
    """Get the global application state."""
    return _state


def get_db_templates() -> Dict[str, List[np.ndarray]]:
    """Get current database templates."""
    return _state.get_db_templates()
